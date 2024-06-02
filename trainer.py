from tqdm import tqdm
import torch
import math
from model import LLaMA 
import time
import numpy as np

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR 
import pandas as pd
from torchinfo import summary
from fairscale.nn import pipe
from tiktoken import get_encoding

from logger import TrainingLogger

from data import create_dataloader 
from pathlib import Path
import math
from functools import partial
import gc
import yaml

class Trainer: 
    def __init__(self, 
                 max_len_seq=2048, 
                 num_layers=12, 
                 dim_model=4096, 
                 dim_hidden=4096*3, 
                 num_heads=32, 
                 prob_dropout=0.25, 
                 batch_size=8, 
                 val_interval=2000, 
                 total_steps=600000, 
                 val_steps=100,
                 grad_accum_steps=64, 
                 lr_peak=2.5e-4, 
                 weight_decay=0.01, 
                 warmup_steps=0.05,
                 balance=[5, 4, 4, 1], 
                 devices=[torch.device(f'cuda:{i}') for i in range(4)]):

        self.tokenizer = get_encoding('gpt2')
        self.criterion = nn.CrossEntropyLoss() 

        # Model hyperparameters
        self.max_len_seq = max_len_seq
        self.num_layers = num_layers
        self.dim_model = dim_model
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.prob_dropout = prob_dropout
        self.vocab_size = self.tokenizer.n_vocab
        self.batch_size = batch_size

        # Training parameters
        self.val_interval = val_interval
        self.val_steps = val_steps 
        self.total_steps = total_steps
        self.grad_accum_steps = grad_accum_steps 
        self.global_step = 0
        self.lr_peak = lr_peak
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.root_dir = Path(__file__).parent
        self.data_dir = self.root_dir / 'data'
        self.results_dir = self.root_dir / 'results'

        with open('config.yaml', 'r') as file: 
            yaml_file = yaml.safe_load(file)
            wandb_config = yaml_file['wandb']
            self.project_name = wandb_config['project_name']
            self.model_name = wandb_config['model_name']

        self.devices = devices
        self.balance = balance

        self.hyperparams = {
            'max_len_seq': max_len_seq,
            'num_layers': num_layers,
            'dim_model': dim_model,
            'dim_hidden': dim_hidden,
            'num_heads': num_heads,
            'prob_dropout': prob_dropout,
            'batch_size': batch_size,
            'val_interval': val_interval,
            'total_steps': total_steps,
            'val_steps': val_steps,
            'grad_accum_steps': grad_accum_steps,
            'lr_peak': lr_peak,
            'weight_decay': weight_decay,
            'warmup_steps': warmup_steps,
            'balance': balance,
            'devices': devices
        }

        self.logger = TrainingLogger(
            project_name=self.project_name, 
            model_name=self.model_name, 
            hyperparams=self.hyperparams, 
            results_dir=self.results_dir)

        self.eval_dl = create_dataloader(data_dir=self.data_dir, split='eval', block_size=max_len_seq, batch_size=batch_size)
        self.val_dl = create_dataloader(data_dir=self.data_dir, split='val', block_size=max_len_seq, batch_size=batch_size) 
        self.train_dl = create_dataloader(data_dir=self.data_dir, split='train', block_size=max_len_seq, batch_size=batch_size)

        print("Loading model...")
        self.model = LLaMA(vocab_size=self.vocab_size, 
            max_len_seq=max_len_seq, 
            n_layers=num_layers, 
            d_model=dim_model, 
            d_ff=self.dim_hidden, 
            n_heads=num_heads, 
            drop_p=prob_dropout)

        summary(self.model, depth=100)
        
        self.model_wrapper = pipe.Pipe(
            module=nn.Sequential(
                nn.Sequential(self.model.decoder.emb_in, self.model.decoder.dropout),
                *[layer for layer in self.model.decoder.layers],
                nn.Sequential(self.model.decoder.norm_out, self.model.decoder.fc_out)
            ), 
            balance=self.balance, 
            devices=self.devices,
            chunks=self.batch_size) # OOM피할 목적..

        summary(self.model_wrapper, depth=100)

        self.params = [p for p in self.model_wrapper.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(self.params, lr=lr_peak, weight_decay=weight_decay) 
        self.scheduler = LambdaLR(
            self.optimizer, 
            lr_lambda=partial(
                self.lr_lambda, 
                warmup_steps=self.warmup_steps // self.grad_accum_steps, 
                total_steps=self.total_steps // self.grad_accum_steps))
        self.amp_ctx_bfloat16 = torch.cuda.amp.autocast(dtype=torch.bfloat16)


    def lr_lambda(self, step, warmup_steps, total_steps): 
        return min(step / warmup_steps, 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))) 
  
    @torch.no_grad()
    def estimate_loss(self, split):
        self.model_wrapper.eval()
        dl = self.eval_dl if split == 'eval' else self.val_dl
        losses = 0.0
        iterator = iter(dl)
        for i in tqdm(range(self.val_steps), desc=f"loss estimation for {split} split.", leave=False):
            batch: torch.TensorBase = next(iterator) 
            with self.amp_ctx_bfloat16:
                y_hat = self.model_wrapper(batch[:, :-1].to(torch.device('cuda:0')))
                loss = self.criterion(y_hat.permute(0, 2, 1), batch[:, 1:].to(torch.device('cuda:3'))).item()
            losses += loss
        self.model_wrapper.train()
        return losses / self.val_steps

    def train(self):
        best_val_loss = float('inf')
        self.global_step = 0
    
        self.optimizer.zero_grad()

        iterator = iter(self.train_dl) 
        train_losses = 0.0
        iter_loss = 0.0
        while self.global_step < self.total_steps:
            start_time = time.time()
            batch: torch.TensorBase = next(iterator)
            with self.amp_ctx_bfloat16:
                y_hat = self.model_wrapper(batch[:, :-1].to(torch.device('cuda:0')))
                loss = self.criterion(y_hat.permute(0, 2, 1), batch[:, 1:].to(torch.device('cuda:3')))
                loss.backward()
                iter_loss = loss.item()
                train_losses += iter_loss

                del batch, y_hat, loss
                gc.collect() 
                torch.cuda.empty_cache()

                if (self.global_step + 1) % self.grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                end_time = time.time()
                iter_time = end_time - start_time
                vram_usage_str = self.logger.get_vram_usage_str()

                print(f"step: {self.global_step:>6}, "
                    f"iter loss: {iter_loss:>7.5f}, "
                    f"grad_accum_step: {self.global_step % self.grad_accum_steps:>2}, "
                    f"iter_times: {iter_time:>5.2f} s, "
                    f"vram: {vram_usage_str}, "
                    f"lr: {self.optimizer.param_groups[0]['lr']:>.5e}") 

                
                if (self.global_step + 1) % self.val_interval == 0:
                    train_loss = train_losses / self.val_interval
                    val_loss = self.estimate_loss('val')
            
                    self.logger.log_metrics(current_steps=self.global_step, train_loss=train_loss, val_loss=val_loss, lr=self.optimizer.param_groups[0]['lr'])
            
                    if val_loss < best_val_loss:
                        self.logger.save_checkpoint(model=self.model, current_steps=self.global_step, max_steps=self.total_steps, optimizer=self.optimizer, scheduler=self.scheduler)
                        best_val_loss = val_loss
            
            self.global_step += 1