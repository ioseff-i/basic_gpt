import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, 
                                     allowed_special = {'<|endoftext|>'}
                                    )
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(
                torch.tensor(
                    token_ids[i:i+max_length]
                )
            )
            self.target_ids.append(
                torch.tensor(
                    token_ids[i+1:i+max_length+1]
                )
            )
            
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
def create_loader_v1(txt,
                     batch_size = 4,
                     max_length = 256,
                     stride = 128,
                     shuffle = True,
                     drop_last = True,
                     num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataset = DataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle = shuffle, 
        drop_last = drop_last,
        num_workers = num_workers
    )
    