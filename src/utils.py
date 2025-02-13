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

def generate(
    model, 
    idx, 
    max_new_tokens, 
    context_size, 
    temperature = 0.0,
    top_k = None,
    eos_id = None):
    
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        if top_k is not None:
            top_logits,_ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits<min_val, 
                                 torch.tensor(float("-inf")).to(logits.device),
                                 logits)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1) #(Batch, 1)
        else:
            idx_next = torch.argmax(logits, dim = -1, keepdim = True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim = 1)
    return idx
def plot_losses(epochs_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("plots/loss-plot.pdf")
    plt.close()