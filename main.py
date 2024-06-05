import argparse
import torch
import random
from trainer import Trainer
import yaml
from data import DataPreprocessor
import numpy as np


def prep_data(config):
    data_pre_processor = DataPreprocessor(
        num_proc=config['num_proc'],
        encoding=config['encoding'],
        train_file=config['train_file'],
        val_file=config['val_file'],
        eval_file=config['eval_file'],
        num_shards=config['num_shards'],
        dataset_name=config['dataset_name'],
        split_ratio=config['split_ratio'],
        seed=config['seed'],
        dtype=eval(config['dtype'])
    )
    data_pre_processor.run()


def train(config):
    config['devices'] = [torch.device(device) for device in config['devices']]
    print("\nHyperparameters Configuration:")
    print("=" * 30)
    for key, value in config.items():
        print(f"{key}: {value}")
    print()

    random_seed = config['random_seed']
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)

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
        devices=config['devices'],
        chunks=config['chunks']
    )
    trainer.train()


def main():
    parser = argparse.ArgumentParser(
        description="Run data preprocessing or training")

    parser.add_argument("task",
                        choices=["prep_data", "train"],
                        help="Task to run: 'prep_data' or 'train'")
    args = parser.parse_args()

    with open('config.yaml', 'r') as file:
        yaml_file = yaml.safe_load(file)

    if args.task == "prep_data":
        prep_data(yaml_file['prep_data'])
    elif args.task == "train":
        train(yaml_file['train'])


if __name__ == "__main__":
    main()