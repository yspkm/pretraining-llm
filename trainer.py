from tqdm import tqdm
import torch
import math
from model import GPT, GPTConfig 
import numpy as np
import time

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
from torchinfo import summary
from tiktoken import get_encoding

from logger import TrainingLogger

from data import create_dataloader
from pathlib import Path
import math
from functools import partial
import gc
import yaml

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


class Trainer:
    def __init__(self, max_len_seq, num_layers, dim_model, dim_hidden, num_heads, prob_dropout, batch_size,
                 val_interval, total_steps, val_steps, grad_accum_steps, lr_peak, weight_decay, warmup_steps):

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

        import os
        self.ddp = int(os.environ.get('RANK', -1)) != -1
        self.init_ddp()
        self.set_seeds()
        torch.cuda.set_device(self.ddp_local_rank)

        self.root_dir = Path(__file__).parent
        self.data_dir = self.root_dir / 'data'
        self.results_dir = self.root_dir / 'results'

        with open('config.yaml', 'r') as file:
            yaml_file = yaml.safe_load(file)
            wandb_config = yaml_file['wandb']
            self.project_name = wandb_config['project_name']
            self.model_name = wandb_config['model_name']

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
        }

        if self.master_process:
            self.logger = TrainingLogger(
                project_name=self.project_name,
                model_name=self.model_name,
                hyperparams=self.hyperparams,
                results_dir=self.results_dir)
        if self.master_process:
            print("Loading model...")
        self.model_config = GPTConfig(block_size=self.max_len_seq, 
                                    vocab_size=self.vocab_size, 
                                    n_layer=self.num_layers, 
                                    n_head=self.num_heads, 
                                    n_embd=self.dim_model, 
                                    dropout=self.prob_dropout, 
                                    bias=True)
        self.model = GPT(config=self.model_config).to(self.device)
        self.raw_model = self.model
        self.model = DDP(self.model, device_ids=[self.ddp_local_rank])
        if self.master_process:
            summary(self.model, depth=100)

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(self.params, lr=lr_peak, weight_decay=weight_decay)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=partial(
                self.lr_lambda,
                warmup_steps=self.warmup_steps,
                total_steps=self.total_steps))
        self.amp_ctx_bfloat16 = torch.cuda.amp.autocast(dtype=torch.bfloat16)

    def get_batch(self, split):
        data_file = 'train.bin' if split == 'train' else 'val.bin'
        data = np.memmap(os.path.join(self.data_dir, data_file), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - self.max_len_seq, (self.batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+self.max_len_seq]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+self.max_len_seq]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x, y


    def lr_lambda(self, step, warmup_steps, total_steps):
        return min(step / warmup_steps,
                   0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps))))

    def init_ddp(self):
        if self.ddp:
            init_process_group(backend='nccl')
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])

            self.device = f'cuda:{self.ddp_local_rank}'
            self.master_process = self.ddp_rank == 0
            self.seed_offset = self.ddp_rank
            assert self.grad_accum_steps % self.ddp_world_size == 0
            self.grad_accum_steps //= self.ddp_world_size
        else:
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1

    def set_seeds(self):
        torch.manual_seed(1337 + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    @torch.no_grad()
    def estimate_loss(self, split):
        self.model.eval()
        dl = self.eval_dl if split == 'eval' else self.val_dl
        losses = 0.0
        for i in tqdm(range(self.val_steps), desc=f"loss estimation for {split} split.", leave=False):
            x, y = self.get_batch('val')
            with self.amp_ctx_bfloat16:
                y_hat = self.model(x)
                loss = self.criterion(y_hat.permute(0, 2, 1), y).item()
            losses += loss
        self.model.train()
        return losses / self.val_steps

    def train(self):
        best_val_loss = float('inf')
        self.global_step = 0

        self.optimizer.zero_grad()

        while self.global_step < self.total_steps:
            start_time = time.time()
            iter_loss = 0.0
            with self.amp_ctx_bfloat16:
                # Gradient Accumulation Steps
                for micro_step in range(self.grad_accum_steps):
                    self.model.require_backward_grad_sync = (micro_step == self.grad_accum_steps -1)
                    x, y = self.get_batch('train')
                    y_hat = self.model(x)
                    loss = self.criterion(y_hat.permute(0, 2, 1), y)
                    loss = loss / self.grad_accum_steps
                    loss.backward()
                    iter_loss += loss.item()

                total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                end_time = time.time()
                iter_time = end_time - start_time
                if self.master_process: 
                    grad_norms = self.logger.calculate_grad_norm(model=self.raw_model)
                    self.logger.log_training(
                        current_steps=self.global_step,
                        train_loss=iter_loss,
                        lr=self.optimizer.param_groups[0]['lr'],
                        iter_time=iter_time,
                        total_grad_norm=total_grad_norm,
                        grad_norms=grad_norms)

                if self.master_process and (self.global_step + 1) % self.val_interval == 0:
                    val_loss = self.estimate_loss('val')
                    self.logger.log_validation(current_steps=self.global_step, val_loss=val_loss)

                    if val_loss < best_val_loss:
                        self.logger.save_checkpoint(model=self.model, current_steps=self.global_step,
                                                    max_steps=self.total_steps, optimizer=self.optimizer,
                                                    scheduler=self.scheduler)
                        best_val_loss = val_loss

            self.global_step += 1

        destroy_process_group()