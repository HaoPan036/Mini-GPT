# 🚀 Getting Started with Mini-GPT

## Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- tqdm (progress bars)
- numpy (numerical operations)

### Step 2: Train the Model

```bash
python3 train.py
```

**What to expect:**
- Training will run for 5,000 iterations (~10-30 minutes on CPU)
- You'll see progress updates every 500 iterations
- Loss should decrease from ~4.0 to ~1.5
- A checkpoint file `ckpt.pt` will be saved

**Example output:**
```
Loading data...
Dataset length: 1115394 characters
Vocabulary size: 65
Model parameters: 10,788,929

Starting training...
Step 0: train loss 4.1745, val loss 4.1692
Step 500: train loss 1.9856, val loss 2.0134
...
```

### Step 3: Generate Text

```bash
python3 generate.py --prompt "ROMEO:" --max_new_tokens 200
```

**Try different settings:**
```bash
# More creative (higher temperature)
python3 generate.py --prompt "To be or not to be" --temperature 1.2

# More focused (lower temperature)
python3 generate.py --prompt "Hello" --temperature 0.5

# Different length
python3 generate.py --prompt "JULIET:" --max_new_tokens 500

# Random generation (no prompt)
python3 generate.py --max_new_tokens 300
```

## Understanding the Parameters

### Training Parameters (in config.py)

- `batch_size`: Number of sequences per training step (default: 64)
- `block_size`: Context length in characters (default: 256)
- `max_iters`: Total training iterations (default: 5000)
- `learning_rate`: How fast the model learns (default: 3e-4)
- `n_embd`: Model embedding dimension (default: 384)
- `num_heads`: Number of attention heads (default: 6)
- `n_layer`: Number of transformer blocks (default: 6)

### Generation Parameters

- `--prompt`: Starting text (optional)
- `--max_new_tokens`: How many characters to generate (default: 200)
- `--temperature`: Randomness (0.1=conservative, 2.0=creative, default: 0.8)
- `--top_k`: Consider only top K most likely tokens (default: 200)

## Project Files Explained

```
mini-gpt/
│
├── data/input.txt          # Training dataset (Shakespeare)
│
├── utils.py                # Character tokenizer
├── config.py               # Hyperparameters
├── model.py                # GPT architecture
├── train.py                # Training script
├── generate.py             # Generation script
│
├── README.md               # Main documentation
├── TROUBLESHOOTING.md      # Common issues
├── FUTURE_WORK.md          # Ideas for improvements
├── CHECKLIST.md            # Verification checklist
└── PROJECT_SUMMARY.md      # Project overview
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

Install PyTorch:
```bash
pip install torch
```

### "CUDA out of memory"

Edit `config.py` and reduce:
```python
batch_size = 32  # was 64
block_size = 128  # was 256
```

### Training is slow

- This is normal on CPU (10-30 minutes)
- Use GPU if available (much faster)
- Or reduce `max_iters` in `config.py` for testing

### Generated text is nonsensical

- Train for more iterations
- Check that training loss is decreasing
- Try lower temperature (0.5-0.7)

## Customization Ideas

### Train on Different Data

1. Replace `data/input.txt` with your own text
2. Run `python3 train.py`
3. The tokenizer will automatically adapt

### Adjust Model Size

Edit `config.py`:
```python
# Smaller model (faster, less capable)
n_embd = 256
n_layer = 4
num_heads = 4

# Larger model (slower, more capable)
n_embd = 512
n_layer = 8
num_heads = 8
```

### Change Training Duration

Edit `config.py`:
```python
# Quick test (2-5 minutes)
max_iters = 1000

# Longer training (better results)
max_iters = 10000
```

## Expected Results

After training for 5,000 iterations, you should see:
- Final train loss: ~1.3-1.5
- Final val loss: ~1.5-1.7
- Generated text that:
  - Has proper spelling
  - Follows Shakespeare-like patterns
  - Forms coherent (though not always meaningful) sentences

## Next Steps

1. ✅ **Complete the training** - Let it run to completion
2. ✅ **Experiment with generation** - Try different prompts and temperatures
3. ✅ **Review the code** - Understand how Transformers work
4. ✅ **Share your results** - Post on GitHub and LinkedIn
5. ✅ **Extend the project** - See FUTURE_WORK.md for ideas

## Learning Resources

- Read `README.md` for detailed architecture explanation
- Check `model.py` to understand the Transformer implementation
- Review `train.py` to see the training loop
- Explore `generate.py` to understand text generation

## Getting Help

If you encounter issues:

1. Check `TROUBLESHOOTING.md` for common problems
2. Review error messages carefully
3. Verify all files exist with `ls -la`
4. Check Python version with `python3 --version` (need 3.7+)
5. Ensure dependencies installed with `pip list | grep torch`

## Success Checklist

- [ ] Dependencies installed successfully
- [ ] Training runs without errors
- [ ] `ckpt.pt` file created
- [ ] Loss decreases during training
- [ ] Generation produces text
- [ ] Can control generation with parameters

Once all checked, you're done! 🎉

---

**Ready to start?**

```bash
pip install -r requirements.txt && python3 train.py
```

**Questions?** Check the other documentation files or review the code comments.

Good luck! 🚀
