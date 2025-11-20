class CharTokenizer:
    """Character-level tokenizer for text processing."""
    
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s):
        """Convert string to list of integers."""
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        """Convert list of integers to string."""
        return "".join([self.itos[i] for i in ids])
