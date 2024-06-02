import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class BinaryDataset(Dataset):
    def __init__(self, data_dir, split, block_size):
        self.data_path = os.path.join(data_dir, f'{split}.bin')
        self.data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.num_blocks = (len(self.data) - 1) // block_size

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        if idx >= self.num_blocks:
            raise IndexError(f"Index {idx} out of range for dataset with {self.num_blocks} blocks.")
        start = idx * self.block_size
        end = start + self.block_size
        return torch.from_numpy(self.data[start:end+1].astype(np.int64)) # 뭉텅이로 보냄

def create_dataloader(data_dir, split, block_size, batch_size):
    dataset = BinaryDataset(data_dir, split, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader