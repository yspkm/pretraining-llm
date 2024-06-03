import logging
from datetime import datetime
from pathlib import Path
import wandb
from torch.utils.tensorboard import SummaryWriter
import torch
import pynvml
import math
from typing import Dict


class TrainingLogger:
    def __init__(self, project_name, model_name, hyperparams, results_dir, notes=""):
        self.run_name = self.generate_run_name(project_name, model_name, hyperparams, notes)
        self.run_dir = self.create_run_directory(results_dir)
        self.logger = self.setup_logger(self.run_dir / f"{self.run_name}.log")
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.init_wandb(project_name, self.run_name, hyperparams, self.run_dir, notes)
        self.checkpoint_path = self.run_dir / 'checkpoint.pt'
        pynvml.nvmlInit()

    @staticmethod
    def generate_run_name(project_name, model_name, hyperparams, notes=""):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{project_name}_{current_time}_{model_name}"
        if notes:
            run_name += f"_{notes}"
        return run_name

    @staticmethod
    def setup_logger(log_file):
        logger = logging.getLogger('training_logger')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            logger.addHandler(fh)
            logger.addHandler(ch)

        return logger

    @staticmethod
    def init_wandb(project_name, run_name, hyperparams, dir, notes=""):
        wandb.init(project=project_name, name=run_name, config=hyperparams, dir=dir, notes=notes)

    def create_run_directory(self, results_dir) -> Path:
        run_dir = Path(results_dir) / self.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def log_validation(self, current_steps, val_loss):
        self.logger.info(
            f"step: {current_steps}, "
            f"loss/val: {val_loss:6.5f}, "
            f"ppl/val: {math.exp(val_loss):10.5f}")

        wandb.log({"loss/val": val_loss,
                   "ppl/val": math.exp(val_loss)}, step=current_steps)

        self.writer.add_scalar('loss/val', val_loss, current_steps)
        self.writer.add_scalar('ppl/val', math.exp(val_loss), current_steps)

    def log_training(self, current_steps, train_loss, lr, iter_time, total_grad_norm, grad_norms):
        # import time
        # start_time = time.time()

        vram_usage = self.get_vram_usage()
        vram_usage_str = ", ".join([f"{key}: {value:.1f}GB" for key, value in vram_usage.items()])

        self.logger.info(
            f"step: {current_steps}, "
            f"iter_time: {iter_time:.3f}, "
            f"learning_rate: {lr:.3e}, "
            f"loss/train: {train_loss:6.5f}, "
            f"ppl/train: {math.exp(train_loss):10.5f}, "
            f"total_grad_norm: {total_grad_norm:.5e}, "
            f"vram_usage: {vram_usage_str}")

        wandb.log(data={"loss/train": train_loss,
                        "iter_time": iter_time,
                        "learning_rate": lr,
                        "ppl/train": math.exp(train_loss),
                        **{f"vram/{key}": value for key, value in vram_usage.items()},
                        **{f"grad_norm/{key}": value for key, value in grad_norms.items()},
                        "grad_norm/total": total_grad_norm},
                  step=current_steps)

        self.writer.add_scalar(tag='iter_time', scalar_value=iter_time, global_step=current_steps)
        self.writer.add_scalar(tag='learning_rate', scalar_value=lr, global_step=current_steps)
        self.writer.add_scalar(tag='loss/train', scalar_value=train_loss, global_step=current_steps)
        self.writer.add_scalar(tag='ppl/train', scalar_value=math.exp(train_loss), global_step=current_steps)
        self.writer.add_scalar(tag='grad_norm/total', scalar_value=total_grad_norm, global_step=current_steps)
        for key, value in vram_usage.items():
            self.writer.add_scalar(tag=f'vram/{key}', scalar_value=value, global_step=current_steps)
        for key, value in grad_norms.items():
            self.writer.add_scalar(tag=f'grad_norm/{key}', scalar_value=value, global_step=current_steps)

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Logging time: {elapsed_time:.3f} seconds") # 0.08 ~ 0.14 s,

    @staticmethod
    def get_vram_usage() -> Dict[str, float]:
        vram_usage = {f"gpu{i}": pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used / 1024 ** 3
                      for i in range(pynvml.nvmlDeviceGetCount())}
        return vram_usage

    @staticmethod
    def calculate_grad_norm(model):
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = torch.norm(param.grad).item()
        return grad_norms

    def save_checkpoint(self, model, current_steps, max_steps, optimizer, scheduler):
        torch.save({
            'state_dict': model.state_dict(),
            'current_steps': current_steps,
            'max_steps': max_steps,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, self.checkpoint_path)

    def __del__(self):
        pynvml.nvmlShutdown()