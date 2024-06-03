import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from tqdm import tqdm
from datasets import load_dataset
import tiktoken
import yaml
from pathlib import Path


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
        return torch.from_numpy(self.data[start:end + 1].astype(np.int64))


def create_dataloader(data_dir, split, block_size, batch_size):
    dataset = BinaryDataset(data_dir, split, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def save_tokenized_data(dset, filename, arr_len, num_shards, dtype):
    """
    Save tokenized dataset to a binary file using numpy.memmap.
    """
    try:
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        idx = 0
        for batch_idx in tqdm(range(num_shards), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=num_shards, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
    except Exception as e:
        print(f"Error writing to {filename}: {e}")


class DataPreprocessor:
    def __init__(self, 
                 num_proc, encoding, train_file, val_file, eval_file, num_shards, dataset_name, split_ratio, seed,
                 dtype):
        self.root_dir = Path(__file__).parent
        self.data_dir = self.root_dir / 'data'

        self.num_proc = num_proc
        self.encoding = encoding
        self.train_file = self.data_dir / train_file
        self.val_file = self.data_dir / val_file
        self.eval_file = self.data_dir / eval_file
        self.num_shards = num_shards
        self.dataset_name = dataset_name
        self.split_ratio = split_ratio
        self.seed = seed
        self.dtype = dtype

        self.enc = tiktoken.get_encoding(self.encoding)

    def process(self, example):
        """
        Tokenize a single example using the specified BPE tokenizer.
        """
        ids = self.enc.encode_ordinary(example['text'])
        ids.append(self.enc.eot_token)  # Add the end of text token
        return {'ids': ids, 'len': len(ids)}

    def run(self):
        dataset = load_dataset(self.dataset_name, num_proc=self.num_proc)

        split_dataset = dataset["train"].train_test_split(test_size=self.split_ratio[1] + self.split_ratio[2],
                                                          seed=self.seed, shuffle=True)

        val_eval_split = split_dataset['test'].train_test_split(
            test_size=self.split_ratio[2] / (self.split_ratio[1] + self.split_ratio[2]), seed=self.seed)
        split_dataset['val'] = val_eval_split['train']
        split_dataset['eval'] = val_eval_split['test']
        split_dataset.pop('test')

        tokenized = split_dataset.map(
            lambda example: self.process(example),
            remove_columns=['text'],
            desc="Tokenizing the splits",
            num_proc=self.num_proc,
        )

        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = {
                'train': self.train_file,
                'val': self.val_file,
                'eval': self.eval_file
            }[split]
            save_tokenized_data(dset, filename, arr_len, self.num_shards, self.dtype)
        print("Data processing complete. Binary files saved.")