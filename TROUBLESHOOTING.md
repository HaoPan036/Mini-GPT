# Troubleshooting Guide

Common issues and solutions when working with the Mini-GPT project.

## Git Push Failures

### Issue: Failed to connect to proxy

**Error message:**
```
Failed to connect to 127.0.0.1:10808
fatal: unable to access 'https://github.com/...': Failed to connect to 127.0.0.1 port 10808 after 0 ms: Couldn't connect to server
```

This happens when Git is configured to use a proxy that's not running or not accessible.

### Solution 1: Remove system proxy environment variables

```bash
unset http_proxy
unset https_proxy
unset all_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
unset ALL_PROXY
```

### Solution 2: Remove proxy configuration from shell profile

Edit your shell configuration file (`~/.zshrc`, `~/.bashrc`, or `~/.bash_profile`) and remove or comment out lines like:

```bash
# Remove these lines:
export http_proxy="http://127.0.0.1:10808"
export https_proxy="http://127.0.0.1:10808"
export all_proxy="socks5://127.0.0.1:10808"
```

After editing, reload the configuration:

```bash
source ~/.zshrc  # or source ~/.bashrc
```

### Solution 3: Remove Git proxy configuration

```bash
# Check current Git proxy settings
git config --global --get http.proxy
git config --global --get https.proxy

# Remove proxy settings
git config --global --unset http.proxy
git config --global --unset https.proxy
```

### Solution 4: Force push after resolving divergence

If you have diverged branches between local and remote:

```bash
# Check current status
git status

# Force push (use with caution!)
git push -u origin main --force

# Or, fetch and merge first (safer)
git fetch origin
git merge origin/main
git push -u origin main
```

## Training Issues

### Issue: CUDA out of memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce `batch_size` in `config.py`
- Reduce `block_size` in `config.py`
- Reduce model size (`n_embd`, `n_layer`, or `num_heads`)
- Use CPU instead: set `device = 'cpu'` in `config.py`

### Issue: Training is too slow on CPU

**Solutions:**
- Reduce `max_iters` for faster experimentation
- Reduce model size
- Use a GPU if available
- Reduce `batch_size` or `block_size`

### Issue: Loss is not decreasing

**Possible causes and solutions:**
- **Learning rate too high**: Reduce `learning_rate` in `config.py` (try 1e-4 or 1e-5)
- **Learning rate too low**: Increase `learning_rate` (try 1e-3)
- **Model too small**: Increase `n_embd` or `n_layer`
- **Data issues**: Check that `data/input.txt` is loaded correctly
- **Gradient issues**: Add gradient clipping in `train.py`:
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  ```

## Generation Issues

### Issue: Generated text is nonsensical

**Solutions:**
- Train for more iterations
- Check that the model checkpoint loaded correctly
- Adjust `temperature`:
  - Lower (0.5-0.8) for more coherent text
  - Higher (1.0-1.5) for more creative text
- Adjust `top_k`:
  - Lower (50-100) for more focused sampling
  - Higher (200-500) for more diversity

### Issue: Generated text is repetitive

**Solutions:**
- Increase `temperature` (try 1.0 or higher)
- Increase `top_k` (try 200-500)
- Train on more diverse data
- Ensure model has enough capacity (`n_layer`, `n_embd`)

### Issue: Model checkpoint not found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'ckpt.pt'
```

**Solution:**
- Run `python train.py` first to create the checkpoint
- Ensure you're in the correct directory
- Specify the correct path: `python generate.py --checkpoint /path/to/ckpt.pt`

## Installation Issues

### Issue: PyTorch installation fails

**Solution:**
Visit [pytorch.org](https://pytorch.org) and use the installation command for your system.

For CPU only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA 11.8:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Import errors

**Error:**
```
ModuleNotFoundError: No module named 'XXX'
```

**Solution:**
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install torch tqdm numpy
```

## Data Issues

### Issue: Dataset file missing

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/input.txt'
```

**Solution:**
Download the Tiny Shakespeare dataset:
```bash
mkdir -p data
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Performance Optimization

### Tips for faster training:

1. **Use GPU**: Set `device = 'cuda'` if available
2. **Increase batch size**: Larger batches are more efficient (if memory allows)
3. **Reduce evaluation frequency**: Increase `eval_interval` in `config.py`
4. **Reduce eval iterations**: Decrease `eval_iters` in `config.py`
5. **Use mixed precision training**: Add to `train.py`:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

### Tips for faster generation:

1. **Use GPU**: Specify `--device cuda`
2. **Reduce max_new_tokens**: Generate fewer tokens
3. **Implement KV caching**: Cache key/value states (see FUTURE_WORK.md)

## Debugging Tips

### Enable detailed error messages:

```bash
export PYTHONVERBOSE=1
python train.py
```

### Check model summary:

Add to `train.py` after model initialization:
```python
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Verify data loading:

```python
# In train.py, after loading data
print(f"First 100 characters: {text[:100]}")
print(f"Sample encoded: {tokenizer.encode('Hello')}")
print(f"Sample decoded: {tokenizer.decode([20, 21, 22])}")
```

### Check gradients:

```python
# In training loop
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
```

## Getting Help

If you encounter issues not covered here:

1. Check the GitHub Issues page
2. Review the code comments in `model.py`, `train.py`, and `generate.py`
3. Compare with the original nanoGPT implementation
4. Enable verbose logging and share error messages when asking for help

---

**Remember:** Most issues can be resolved by:
- Checking file paths and working directory
- Verifying dependencies are installed
- Reducing model/batch size if running out of memory
- Training longer if results are poor
