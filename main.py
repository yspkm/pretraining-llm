import torch
import random
from trainer import Trainer
import yaml 

random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(random_seed)

with open('config.yaml', 'r') as file:
    yaml_file = yaml.safe_load(file)
    config = yaml_file['config']
config['devices'] = [torch.device(device) for device in config['devices']]
print("\nHyperparameters Configuration:")
print("=" * 30)
for key, value in config.items():
    print(f"{key}: {value}")
print()

trainer = Trainer(
    max_len_seq=config['max_len_seq'],
    num_layers=config['num_layers'],
    dim_model=config['dim_model'],
    dim_hidden=config['dim_hidden'],
    num_heads=config['num_heads'],
    prob_dropout=config['prob_dropout'],
    batch_size=config['batch_size'],
    val_interval=config['val_interval'],
    total_steps=config['total_steps'],
    val_steps=config['val_steps'],
    grad_accum_steps=config['grad_accum_steps'],
    lr_peak=config['lr_peak'],
    weight_decay=config['weight_decay'],
    warmup_steps=config['warmup_steps'],
    balance=config['balance'],
    devices=config['devices']
)

trainer.train()