import logging
from datetime import datetime
from pathlib import Path
import wandb
from torch.utils.tensorboard import SummaryWriter
import torch
import pynvml

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
        #hyperparams_str = "_".join([f"{key}{value}" for key, value in hyperparams.items()])
        #run_name = f"{project_name}_{current_time}_{model_name}_{hyperparams_str}"
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

    def log_metrics(self, current_steps, train_loss, val_loss, lr):
        self.logger.info(f"current steps: {current_steps}: train loss: {train_loss:.5f}   val loss: {val_loss:.5f}   learning rate: {lr:.8f}")
        wandb.log({"current_steps": current_steps, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": lr})
        self.writer.add_scalar('loss/train', train_loss, current_steps)
        self.writer.add_scalar('loss/val', val_loss, current_steps)
        self.writer.add_scalar('learning_rate', lr, current_steps)

    def wandb_log_cur_step_only(self, current_steps, train_loss, lr, iter_time, grad_norm):
        wandb.log({"current_steps": current_steps, "train_loss": train_loss, "learning_rate": lr, 'iter_time': iter_time, "grad_norm": grad_norm})
    
    def get_vram_usage_str(self)->str:
        vram_usage = {f"GPU{i}": pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used / 1024 ** 3
                      for i in range(pynvml.nvmlDeviceGetCount())}
        vram_usage_str = ", ".join([f"{v:.1f}GB" for v in vram_usage.values()])
        return vram_usage_str

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
