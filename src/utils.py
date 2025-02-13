import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())
    
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
def create_dataloader_v1(txt,
                     batch_size = 4,
                     max_length = 256,
                     stride = 128,
                     shuffle = True,
                     drop_last = True,
                     num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    return DataLoader(
        dataset, 
        batch_size = batch_size,
        shuffle = shuffle, 
        drop_last = drop_last,
        num_workers = num_workers
    )

def generate_text_simple(model, idx, max_new_tokens,context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim = -1, keepdim = True)    #(Batch, 1)
        idx = torch.cat((idx, idx_next), dim = 1) #(Batch, n_tokens = 1)
    return idx
    