import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from model import GPTModel
from utils import CharTokenizer
from config import *


# Load data
print("Loading data...")
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset length: {len(text)} characters")

# Initialize tokenizer
tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

# Encode entire dataset
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
print(f"Encoded data shape: {data.shape}")

# Train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """Generate a small batch of data of inputs x and targets y."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and val sets."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Initialize model
print(f"\nInitializing model on {device}...")
model = GPTModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    num_heads=num_heads,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
)
model = model.to(device)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
print(f"\nStarting training for {max_iters} iterations...")
print(f"Batch size: {batch_size}, Block size: {block_size}")
print("-" * 60)

for iter in tqdm(range(max_iters), desc="Training"):
    # Evaluate loss periodically
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"\nStep {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # Sample a batch of data
    xb, yb = get_batch('train')
    
    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n" + "=" * 60)
print("Training complete!")

# Save model
print("Saving model checkpoint...")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'tokenizer_stoi': tokenizer.stoi,
    'tokenizer_itos': tokenizer.itos,
    'vocab_size': vocab_size,
    'config': {
        'n_embd': n_embd,
        'num_heads': num_heads,
        'n_layer': n_layer,
        'block_size': block_size,
        'dropout': dropout
    }
}, 'ckpt.pt')

print("Checkpoint saved to ckpt.pt")
print("\nGenerating sample text...")

# Generate sample
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=200)
print("\n" + "=" * 60)
print("GENERATED TEXT:")
print("=" * 60)
print(tokenizer.decode(generated[0].tolist()))
print("=" * 60)
