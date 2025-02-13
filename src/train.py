from importlib.metadata import version

import matplotlib
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..')))
import argparse
from model import GPTModel
from utils import *


GPT_CONFIG_124M = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

torch.manual_seed(123)
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(
        logits.flatten(0,1),
        target_batch.flatten()
    )
    return loss
def calc_loss_loader(data_loader, model, device, num_batches = None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i,(input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches



import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    early_stopping_patience=3,  # Number of epochs to wait for improvement
    min_lr=2e-5,  # Minimum learning rate for cosine scheduler
):
    # Initialize lists to track losses
    train_losses, val_losses = [], []
    global_step = 0

    # Initialize cosine learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    # Early stopping variables
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Training loop
    for epoch in trange(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # Gradient clipping
            max_norm = 1.0  # Set the maximum norm for gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            global_step += 1

            # Evaluation and logging
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter, device
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                # print(
                #     f"Epoch {epoch}, Global Step {global_step}: "
                #     f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
                # )

                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)."
                    )
                    return train_losses, val_losses

        # Update learning rate using cosine scheduler
        scheduler.step()

        # Generate and print sample text
        # generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses


def evaluate_model(model, train_loader, val_loader, eval_iter, device):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[1]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=64, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()



if __name__=='__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model = GPTModel(GPT_CONFIG_124M).to(device)


    file_path = "data/the-verdict.txt"
    
    # file_path = "../data/fuzuli.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
        

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    
    
    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    tokenizer = tiktoken.get_encoding("gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    
    num_epochs = 100
    # Train the model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        early_stopping_patience=5,
        min_lr=6e-6,
    )
    
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, train_losses, val_losses)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4)

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    
    




