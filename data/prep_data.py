import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import tiktoken

NUM_PROC = 64  # Number of workers for .map() and load_dataset() calls
ENCODING = "gpt2"
TRAIN_FILE = 'train.bin'
VAL_FILE = 'val.bin'
EVAL_FILE = 'eval.bin'
NUM_SHARDS = 1024

enc = tiktoken.get_encoding(ENCODING)

def process(example):
    """
    Tokenize a single example using GPT-2 BPE tokenizer.
    """
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)  # Add the end of text token
    return {'ids': ids, 'len': len(ids)}

def save_tokenized_data(dset, filename, arr_len, num_shards=NUM_SHARDS, dtype=np.uint16):
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

if __name__ == '__main__':
    dataset = load_dataset("openwebtext", num_proc=NUM_PROC)

    split_ratio = [0.999, 0.0005, 0.0005]
    split_dataset = dataset["train"].train_test_split(test_size=split_ratio[1] + split_ratio[2], seed=2357, shuffle=True)
    
    val_eval_split = split_dataset['test'].train_test_split(test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), seed=2357)
    split_dataset['val'] = val_eval_split['train']
    split_dataset['eval'] = val_eval_split['test']
    split_dataset.pop('test')  # Remove the intermediate 'test' split

    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing the splits",
        num_proc=NUM_PROC,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = {
            'train': TRAIN_FILE,
            'val': VAL_FILE,
            'eval': EVAL_FILE
        }[split]
        save_tokenized_data(dset, filename, arr_len)
        
    print("Data processing complete. Binary files saved.")