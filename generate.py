import torch
import argparse

from model import GPTModel
from utils import CharTokenizer


def load_model(checkpoint_path='ckpt.pt'):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config
    config = checkpoint['config']
    vocab_size = checkpoint['vocab_size']
    
    # Reconstruct tokenizer
    class SimpleTokenizer:
        def __init__(self, stoi, itos):
            self.stoi = stoi
            self.itos = itos
            self.vocab_size = len(stoi)
        
        def encode(self, s):
            return [self.stoi[ch] for ch in s]
        
        def decode(self, ids):
            return "".join([self.itos[i] for i in ids])
    
    tokenizer = SimpleTokenizer(checkpoint['tokenizer_stoi'], checkpoint['tokenizer_itos'])
    
    # Reconstruct model
    model = GPTModel(
        vocab_size=vocab_size,
        n_embd=config['n_embd'],
        num_heads=config['num_heads'],
        n_layer=config['n_layer'],
        block_size=config['block_size'],
        dropout=0.0  # No dropout during inference
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully! (vocab_size={vocab_size})")
    return model, tokenizer


def generate_text(model, tokenizer, prompt="", max_new_tokens=200, temperature=1.0, top_k=None, device='cpu'):
    """Generate text from a prompt."""
    model = model.to(device)
    
    # Encode the prompt
    if prompt:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    generated_text = tokenizer.decode(generated[0].tolist())
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text using trained Mini-GPT model')
    parser.add_argument('--prompt', type=str, default='', help='Initial prompt for generation')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k sampling parameter')
    parser.add_argument('--checkpoint', type=str, default='ckpt.pt', help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.checkpoint)
    
    # Generate text
    print("\n" + "=" * 60)
    print("GENERATING TEXT")
    print("=" * 60)
    if args.prompt:
        print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print("=" * 60)
    
    generated_text = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device
    )
    
    print("\n" + generated_text)
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
