# 🛠️ Mini-GPT 实践教程 - 一步一步动手做

## 🎯 学习目标

通过实际操作，深入理解 Mini-GPT 的每个组件。

---

# 第一天：理解分词器（30分钟）

## 步骤1：创建测试文件

```bash
cd /Users/hao/Desktop/CS336/mini-gpt
```

## 步骤2：创建并运行第一个测试

创建文件 `day1_tokenizer.py`：

```python
# ===== 实验1：基础分词器测试 =====
from utils import CharTokenizer

# 简单的英文文本
text = "hello world"
print("=" * 50)
print("实验1：英文分词器")
print("=" * 50)
print(f"原始文本: {text}")

# 创建分词器
tokenizer = CharTokenizer(text)

# 查看词汇表
print(f"\n词汇表大小: {tokenizer.vocab_size}")
print(f"所有字符: {sorted(tokenizer.stoi.keys())}")
print(f"\n字符→数字映射:")
for char, idx in sorted(tokenizer.stoi.items()):
    print(f"  '{char}' → {idx}")

# 测试编码
test_word = "hello"
encoded = tokenizer.encode(test_word)
print(f"\n编码 '{test_word}': {encoded}")

# 测试解码
decoded = tokenizer.decode(encoded)
print(f"解码回来: '{decoded}'")

# 验证正确性
print(f"\n验证: {test_word == decoded} ✓" if test_word == decoded else f"✗ 错误！")

print("\n" + "=" * 50)
print("实验2：中英文混合")
print("=" * 50)

# 中英文混合
mixed_text = "Hello 你好"
tokenizer2 = CharTokenizer(mixed_text)
print(f"原始文本: {mixed_text}")
print(f"词汇表大小: {tokenizer2.vocab_size}")
print(f"字符集: {sorted(tokenizer2.stoi.keys())}")

encoded_mixed = tokenizer2.encode(mixed_text)
decoded_mixed = tokenizer2.decode(encoded_mixed)
print(f"\n编码: {encoded_mixed}")
print(f"解码: '{decoded_mixed}'")

print("\n" + "=" * 50)
print("实验3：数据集统计")
print("=" * 50)

# 加载实际数据集
with open('data/input.txt', 'r', encoding='utf-8') as f:
    shakespeare = f.read()

print(f"莎士比亚文本长度: {len(shakespeare):,} 字符")
tokenizer3 = CharTokenizer(shakespeare)
print(f"唯一字符数: {tokenizer3.vocab_size}")
print(f"前100个字符: {shakespeare[:100]}")

# 编码前10个字符
sample = shakespeare[:10]
sample_encoded = tokenizer3.encode(sample)
print(f"\n样本: '{sample}'")
print(f"编码: {sample_encoded}")
print(f"解码: '{tokenizer3.decode(sample_encoded)}'")
```

**运行**：
```bash
python3 day1_tokenizer.py
```

## 步骤3：思考与记录

在笔记本记录：
1. 词汇表大小对什么有影响？
2. 为什么中文字符和英文字符混合时词汇表会变大？
3. 如果遇到训练时没见过的字符会怎样？（试试 `tokenizer.encode("xyz123")`）

---

# 第二天：理解 Embedding（40分钟）

## 步骤1：创建 Embedding 实验

创建文件 `day2_embedding.py`：

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

print("=" * 50)
print("实验1：Token Embedding 基础")
print("=" * 50)

# 创建一个小型 embedding
vocab_size = 10  # 10个不同的token
embed_dim = 4    # 每个token用4维向量表示

embedding = nn.Embedding(vocab_size, embed_dim)

# 查看embedding矩阵
print(f"Embedding 矩阵形状: {embedding.weight.shape}")
print(f"这是一个 {vocab_size} × {embed_dim} 的矩阵\n")

# 测试: 将 token 0 转换成向量
token_0 = torch.tensor([0])
vec_0 = embedding(token_0)
print(f"Token 0 的向量: {vec_0.squeeze().detach().numpy()}")

# 测试: 将多个 tokens 转换
tokens = torch.tensor([0, 1, 2, 0])  # 一个序列
vecs = embedding(tokens)
print(f"\n序列 {tokens.tolist()} 的向量矩阵形状: {vecs.shape}")
print(f"这表示 4个token，每个用4维向量")

print("\n" + "=" * 50)
print("实验2：相似度计算")
print("=" * 50)

# 计算两个token的相似度
vec_0 = embedding(torch.tensor([0])).squeeze()
vec_1 = embedding(torch.tensor([1])).squeeze()

# 余弦相似度
cos_sim = torch.cosine_similarity(vec_0, vec_1, dim=0)
print(f"Token 0 和 Token 1 的余弦相似度: {cos_sim.item():.4f}")
print("相似度越接近1，表示越相似")

print("\n" + "=" * 50)
print("实验3：Position Embedding")
print("=" * 50)

# 位置编码
max_len = 8
pos_embedding = nn.Embedding(max_len, embed_dim)

# 序列的位置
positions = torch.arange(max_len)
pos_vecs = pos_embedding(positions)

print(f"8个位置的编码矩阵形状: {pos_vecs.shape}")
print(f"\n位置0的向量: {pos_vecs[0].detach().numpy()}")
print(f"位置1的向量: {pos_vecs[1].detach().numpy()}")

print("\n" + "=" * 50)
print("实验4：完整输入 = Token + Position")
print("=" * 50)

# 一个例子序列
sequence = torch.tensor([3, 1, 4, 1])
seq_len = len(sequence)

# Token embeddings
token_emb = embedding(sequence)
print(f"Token embeddings 形状: {token_emb.shape}")

# Position embeddings
positions = torch.arange(seq_len)
pos_emb = pos_embedding(positions)
print(f"Position embeddings 形状: {pos_emb.shape}")

# 相加
final_input = token_emb + pos_emb
print(f"最终输入形状: {final_input.shape}")
print("\n这就是输入到 Transformer 的数据！")

# 可视化（如果有 matplotlib）
try:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(token_emb.detach().numpy(), aspect='auto', cmap='coolwarm')
    plt.title('Token Embeddings')
    plt.ylabel('Sequence Position')
    plt.xlabel('Embedding Dimension')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(pos_emb.detach().numpy(), aspect='auto', cmap='coolwarm')
    plt.title('Position Embeddings')
    plt.xlabel('Embedding Dimension')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(final_input.detach().numpy(), aspect='auto', cmap='coolwarm')
    plt.title('Final Input (Token + Position)')
    plt.xlabel('Embedding Dimension')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('day2_embeddings.png')
    print("\n✓ 可视化已保存到 day2_embeddings.png")
except:
    print("\n(可视化需要 matplotlib)")
```

**运行**：
```bash
python3 day2_embedding.py
```

---

# 第三天：理解注意力机制（60分钟）

## 步骤1：手动计算注意力

创建文件 `day3_attention.py`：

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("=" * 60)
print("手把手理解注意力机制")
print("=" * 60)

# ===== 第一部分：最简单的注意力例子 =====
print("\n【第1步】创建一个简单的序列")
print("-" * 60)

# 假设我们有3个词，每个词用2维向量表示
sequence = torch.tensor([
    [1.0, 0.0],  # 词1
    [0.0, 1.0],  # 词2
    [0.5, 0.5],  # 词3
])

print(f"序列形状: {sequence.shape}")  # (3, 2)
print(f"3个词，每个词2维\n")
print("序列内容:")
print(sequence)

# ===== 第二部分：创建 Q, K, V =====
print("\n【第2步】创建 Query, Key, Value")
print("-" * 60)

# 为了简化，我们直接使用序列本身
Q = sequence  # Query: "我要找什么"
K = sequence  # Key: "我有什么"
V = sequence  # Value: "内容是什么"

print("在这个例子中，Q = K = V = sequence")
print("(实际中会用线性层变换)")

# ===== 第三部分：计算注意力分数 =====
print("\n【第3步】计算注意力分数")
print("-" * 60)

# Q @ K^T
scores = Q @ K.transpose(0, 1)
print(f"注意力分数矩阵形状: {scores.shape}")  # (3, 3)
print("\n注意力分数 (Q @ K^T):")
print(scores.numpy())
print("\n解释:")
print("scores[i][j] = 词i对词j的关注程度")

# 缩放
d_k = Q.shape[-1]  # 维度
scores_scaled = scores / (d_k ** 0.5)
print(f"\n缩放后 (除以 √{d_k}):")
print(scores_scaled.numpy())

# Softmax 转换成概率
attention_weights = F.softmax(scores_scaled, dim=-1)
print("\nSoftmax 后 (转换成概率):")
print(attention_weights.numpy())
print("\n每一行的和:", attention_weights.sum(dim=-1).numpy())
print("✓ 每行和为1，这就是概率分布！")

# ===== 第四部分：应用注意力 =====
print("\n【第4步】应用注意力到 Value")
print("-" * 60)

output = attention_weights @ V
print(f"输出形状: {output.shape}")
print("\n输出:")
print(output.numpy())
print("\n这就是注意力的结果！")
print("每个词的输出 = 所有词的加权平均")

# ===== 第五部分：因果掩码 =====
print("\n【第5步】因果掩码 (Causal Mask)")
print("-" * 60)

# 创建因果掩码
mask = torch.tril(torch.ones(3, 3))
print("因果掩码 (下三角矩阵):")
print(mask.numpy())
print("\n1表示可以看，0表示不能看")

# 应用掩码
scores_masked = scores_scaled.masked_fill(mask == 0, float('-inf'))
print("\n应用掩码后:")
print(scores_masked.numpy())
print("\n-inf 的位置在 softmax 后会变成0")

# Softmax
attention_weights_masked = F.softmax(scores_masked, dim=-1)
print("\nSoftmax 后:")
print(attention_weights_masked.numpy())
print("\n观察:")
print("- 第1行: 只能看自己")
print("- 第2行: 可以看前2个词")
print("- 第3行: 可以看所有3个词")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 无掩码的注意力
im1 = axes[0].imshow(attention_weights.numpy(), cmap='Blues', vmin=0, vmax=1)
axes[0].set_title('Attention Weights (No Mask)')
axes[0].set_xlabel('Key Position')
axes[0].set_ylabel('Query Position')
plt.colorbar(im1, ax=axes[0])

# 有掩码的注意力
im2 = axes[1].imshow(attention_weights_masked.numpy(), cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Attention Weights (Causal Mask)')
axes[1].set_xlabel('Key Position')
axes[1].set_ylabel('Query Position')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('day3_attention.png')
print("\n✓ 可视化已保存到 day3_attention.png")

# ===== 第六部分：实际模型的注意力 =====
print("\n" + "=" * 60)
print("用真实模型的注意力头")
print("=" * 60)

from model import Head

# 创建一个注意力头
n_embd = 32
head_size = 8
block_size = 10

head = Head(n_embd, head_size, block_size)

# 创建一个随机输入 (batch=1, seq_len=5, embed_dim=32)
x = torch.randn(1, 5, n_embd)
print(f"\n输入形状: {x.shape}")

# 前向传播
output = head(x)
print(f"输出形状: {output.shape}")
print("\n✓ 注意力头成功运行！")

print("\n" + "=" * 60)
print("总结")
print("=" * 60)
print("""
注意力机制的核心步骤：

1. 输入序列 → Q, K, V (通过线性变换)
2. 计算相似度: Q @ K^T
3. 缩放: / √d_k
4. 应用掩码 (对于因果注意力)
5. Softmax: 转换成概率
6. 加权求和: @ V
7. 输出结果

关键理解：
- 注意力就是"加权平均"
- 权重来自相似度计算
- 因果掩码防止看到未来
""")
```

**运行**：
```bash
python3 day3_attention.py
```

---

# 第四天：训练你的第一个模型（90分钟）

## 步骤1：小规模快速实验

创建文件 `day4_train_small.py`：

```python
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

print("=" * 60)
print("训练一个超小型模型（5分钟内完成）")
print("=" * 60)

# ===== 配置 =====
# 超小参数，快速看到效果
batch_size = 16
block_size = 32
max_iters = 500  # 只训练500步
eval_interval = 100
learning_rate = 1e-3

# 模型参数（很小）
n_embd = 64
num_heads = 2
n_layer = 2
dropout = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# ===== 加载数据 =====
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 使用前10000个字符（更快）
text = text[:10000]
print(f"\n数据长度: {len(text)} 字符")

# 分词器
from utils import CharTokenizer
tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size
print(f"词汇表大小: {vocab_size}")

# 编码
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# 分割
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"训练集: {len(train_data)} tokens")
print(f"验证集: {len(val_data)} tokens")

# ===== 数据加载函数 =====
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ===== 创建模型 =====
from model import GPTModel

model = GPTModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    num_heads=num_heads,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
)
model = model.to(device)

# 统计参数
n_params = sum(p.numel() for p in model.parameters())
print(f"\n模型参数: {n_params:,}")

# ===== 优化器 =====
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ===== 评估函数 =====
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(20)  # 少量评估，更快
        for k in range(20):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ===== 训练循环 =====
print("\n" + "=" * 60)
print("开始训练")
print("=" * 60)

losses_history = {'train': [], 'val': []}

for iter in tqdm(range(max_iters), desc="训练中"):
    # 评估
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        losses_history['train'].append(losses['train'])
        losses_history['val'].append(losses['val'])
        print(f"\nStep {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # 训练步骤
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n✓ 训练完成！")

# ===== 生成样本 =====
print("\n" + "=" * 60)
print("生成样本文本")
print("=" * 60)

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=50)
generated_text = tokenizer.decode(generated[0].tolist())

print("\n生成的文本:")
print("-" * 60)
print(generated_text)
print("-" * 60)

# ===== 可视化 loss =====
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(losses_history['train'], label='Train Loss', marker='o')
plt.plot(losses_history['val'], label='Val Loss', marker='s')
plt.xlabel('Evaluation Step')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.savefig('day4_training_curve.png')
print("\n✓ Loss 曲线已保存到 day4_training_curve.png")

# ===== 保存模型 =====
torch.save({
    'model_state_dict': model.state_dict(),
    'tokenizer_stoi': tokenizer.stoi,
    'tokenizer_itos': tokenizer.itos,
    'vocab_size': vocab_size,
    'config': {
        'n_embd': n_embd,
        'num_heads': num_heads,
        'n_layer': n_layer,
        'block_size': block_size,
    }
}, 'day4_model.pt')

print("✓ 模型已保存到 day4_model.pt")

print("\n" + "=" * 60)
print("实验总结")
print("=" * 60)
print(f"""
训练配置:
- 数据量: {len(text)} 字符
- 训练步数: {max_iters}
- 模型参数: {n_params:,}
- 最终训练 loss: {losses_history['train'][-1]:.4f}
- 最终验证 loss: {losses_history['val'][-1]:.4f}

观察:
1. Loss 是否下降？
2. 生成的文本是否有改善？
3. 训练 loss 和验证 loss 的关系？

下一步:
1. 增加训练步数 (max_iters = 2000)
2. 增加模型大小 (n_embd = 128, n_layer = 4)
3. 使用完整数据集
""")
```

**运行**：
```bash
python3 day4_train_small.py
```

这将在5-10分钟内完成！

---

# 第五天：完整训练与实验（自主探索）

## 实验清单

### ✅ 实验1：不同模型大小

修改 `config.py`，尝试：

```python
# 极小模型
n_embd = 64
n_layer = 2
num_heads = 2

# 小模型
n_embd = 128
n_layer = 4
num_heads = 4

# 默认模型
n_embd = 384
n_layer = 6
num_heads = 6
```

**记录**：参数量、训练时间、最终 loss

### ✅ 实验2：不同学习率

```python
learning_rate_options = [1e-5, 3e-4, 1e-3, 3e-3]
```

**观察**：哪个学习率收敛最快？

### ✅ 实验3：生成参数

```bash
# Temperature
python3 generate.py --temperature 0.3  # 保守
python3 generate.py --temperature 1.0  # 平衡
python3 generate.py --temperature 1.5  # 创新

# Top-K
python3 generate.py --top_k 10   # 限制选择
python3 generate.py --top_k 200  # 更多选择
```

**比较**：输出的质量和多样性

### ✅ 实验4：自定义数据集

创建你自己的文本文件：

```bash
# 中文文本
cat > data/chinese.txt << 'END'
床前明月光，疑是地上霜。
举头望明月，低头思故乡。
...
END

# 修改 train.py 中的数据路径
# 训练
python3 train.py
```

---

# 学习检查点

## Week 1 Checkpoint

- [ ] 能解释分词器的作用
- [ ] 理解 stoi 和 itos 的区别
- [ ] 能手动编码/解码一段文字

## Week 2 Checkpoint

- [ ] 理解 Embedding 的概念
- [ ] 知道 Position Embedding 的作用
- [ ] 能画出注意力机制的流程图

## Week 3 Checkpoint

- [ ] 成功运行一次完整训练
- [ ] 理解训练循环的每个步骤
- [ ] 能解释 loss 下降的含义

## Week 4 Checkpoint

- [ ] 尝试过至少3种不同配置
- [ ] 能独立调试常见错误
- [ ] 生成的文本质量可接受

---

# 下一步：进阶项目

## 项目1：可视化工具

创建一个可视化注意力权重的工具

## 项目2：对话模型

训练一个简单的问答模型

## 项目3：代码生成

在代码数据集上训练

## 项目4：性能优化

实现 KV cache，加速生成

---

**记住：实践是最好的老师！动手做，多实验！** 🚀
