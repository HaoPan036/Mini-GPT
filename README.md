# Mini-GPT: A Lightweight GPT-like Model from Scratch in PyTorch

A minimal, educational implementation of a GPT-style decoder-only Transformer built entirely from scratch using PyTorch. Inspired by Andrej Karpathy's nanoGPT, this project demonstrates the core concepts of modern language models in a clean, understandable codebase.

## 🚀 Project Summary

This project implements a character-level language model trained on the Tiny Shakespeare dataset. It includes:
- Complete Transformer architecture (multi-head self-attention, feed-forward networks, residual connections)
- Character-level tokenization
- Full training pipeline with evaluation
- Autoregressive text generation with temperature and top-k sampling
- Model checkpointing and loading

## ✨ Features

- **Decoder-Only Transformer**: Implements the GPT architecture with causal self-attention
- **Multi-Head Attention**: Parallel attention heads with learned Q, K, V projections
- **Positional Encoding**: Learned positional embeddings for sequence modeling
- **Pre-Layer Normalization**: Modern transformer architecture with Pre-LN
- **Residual Connections**: Skip connections for stable training
- **Character-Level Tokenizer**: Simple and interpretable tokenization
- **Autoregressive Generation**: Sample text with temperature and top-k controls
- **Checkpointing**: Save and load trained models

## 📊 Model Architecture

```
GPTModel
├── Token Embedding (vocab_size → n_embd)
├── Position Embedding (block_size → n_embd)
├── Transformer Blocks (× n_layer)
│   ├── LayerNorm
│   ├── Multi-Head Self-Attention
│   │   ├── Q, K, V Projections
│   │   ├── Causal Masking (torch.tril)
│   │   └── Output Projection
│   ├── Residual Connection
│   ├── LayerNorm
│   ├── Feed-Forward Network (n_embd → 4×n_embd → n_embd)
│   └── Residual Connection
├── Final LayerNorm
└── Language Model Head (n_embd → vocab_size)
```

**Default Hyperparameters:**
- Embedding dimension: 384
- Number of layers: 6
- Number of attention heads: 6
- Block size (context length): 256
- Dropout: 0.2

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mini-gpt.git
cd mini-gpt

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.7+
- PyTorch
- tqdm
- numpy

## 🎯 How to Train

1. **Prepare the data**: The Tiny Shakespeare dataset should be in `data/input.txt` (already included)

2. **Start training**:
```bash
python train.py
```

Training progress will be displayed with a progress bar. The model will:
- Evaluate train/val loss every 500 iterations
- Save the final checkpoint to `ckpt.pt`
- Generate a sample text at the end

**Training output example:**
```
Loading data...
Dataset length: 1115394 characters
Vocabulary size: 65
Model parameters: 10,788,929

Starting training for 5000 iterations...
Step 0: train loss 4.1745, val loss 4.1692
Step 500: train loss 1.9856, val loss 2.0134
...
```

## 🎨 How to Generate Text

After training, use `generate.py` to create new text:

```bash
# Generate with default settings
python generate.py

# Generate with a custom prompt
python generate.py --prompt "ROMEO:" --max_new_tokens 300

# Adjust creativity (temperature)
python generate.py --prompt "To be or not to be" --temperature 1.0 --top_k 100

# All available options
python generate.py \
    --prompt "Hello" \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_k 200 \
    --checkpoint ckpt.pt \
    --device cpu
```

**Parameters:**
- `--prompt`: Starting text (empty for random generation)
- `--max_new_tokens`: Number of new tokens to generate
- `--temperature`: Sampling temperature (higher = more random, lower = more deterministic)
- `--top_k`: Consider only top-k most likely tokens
- `--checkpoint`: Path to saved model checkpoint
- `--device`: Use 'cuda' for GPU or 'cpu'

**Example output:**
```
GENERATING TEXT
============================================================
Prompt: ROMEO:
Max tokens: 300
Temperature: 0.8
Top-k: 200
============================================================

ROMEO:
What shall I do, but with the world's consent,
That I should love thee, and be thy friend?
...
```

## 📁 Project Structure

```
mini-gpt/
│
├── data/
│   └── input.txt              # Tiny Shakespeare dataset
│
├── model.py                   # GPT model architecture
├── train.py                   # Training script
├── generate.py                # Text generation script
├── utils.py                   # Character tokenizer
├── config.py                  # Hyperparameters
├── requirements.txt           # Python dependencies
├── ckpt.pt                    # Saved model (after training)
│
├── README.md                  # This file
├── TROUBLESHOOTING.md         # Common issues and solutions
└── FUTURE_WORK.md             # Planned improvements
```

## 🧠 Key Learnings

This project demonstrates several fundamental concepts in modern NLP:

1. **Self-Attention Mechanism**: How tokens "attend" to previous tokens in the sequence using Q, K, V matrices
2. **Causal Masking**: Preventing the model from "looking ahead" during training
3. **Residual Streams**: Skip connections that allow gradients to flow more easily
4. **Layer Normalization**: Stabilizing training by normalizing activations
5. **Autoregressive Generation**: Sampling tokens one at a time, conditioning on previous tokens
6. **Character-Level Modeling**: Understanding text at the character granularity

## 🔮 Future Improvements

See `FUTURE_WORK.md` for detailed plans. Key areas:

- **BPE Tokenizer**: Implement Byte-Pair Encoding for more efficient tokenization
- **Weights & Biases Integration**: Add experiment tracking and visualization
- **Multi-GPU Support**: Distributed training for larger models
- **Chinese Dataset**: Train on Chinese text (e.g., Chinese poetry)
- **KV Caching**: Optimize generation speed by caching key/value states
- **Flash Attention**: More efficient attention computation
- **Learning Rate Scheduling**: Cosine annealing, warmup
- **Gradient Clipping**: Prevent gradient explosion

## 📝 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy's minimal GPT implementation
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide to Transformers

## 📄 License

MIT License - feel free to use this code for learning and experimentation!

---

## 💼 LinkedIn Project Summary

**Copy-friendly short description for LinkedIn:**

```
Mini-GPT: A nanoGPT-style Transformer built entirely from scratch using PyTorch.
Includes tokenizer, training pipeline, autoregressive generation, and full project setup.

✅ Decoder-only Transformer architecture
✅ Multi-head self-attention with causal masking
✅ Character-level tokenization
✅ Training on Tiny Shakespeare dataset
✅ Text generation with temperature & top-k sampling
✅ Complete documentation and reproducible setup

Skills: PyTorch, Transformers, NLP, LLMs, Deep Learning, Python

GitHub: [your-repo-link]
```

---

**Built with ❤️ for learning and understanding language models**
