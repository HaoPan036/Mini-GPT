# 🎓 Mini-GPT 项目完整学习指南

## 📚 目录

1. [第一部分：项目概览与基础知识](#第一部分项目概览与基础知识)
2. [第二部分：数据处理与分词器](#第二部分数据处理与分词器)
3. [第三部分：Transformer 模型架构](#第三部分transformer-模型架构)
4. [第四部分：训练流程](#第四部分训练流程)
5. [第五部分：文本生成](#第五部分文本生成)
6. [第六部分：实践练习](#第六部分实践练习)

---

# 第一部分：项目概览与基础知识

## 1.1 什么是 GPT？

**GPT** = **G**enerative **P**re-trained **T**ransformer（生成式预训练 Transformer）

### 核心概念：
- **生成式**：模型可以生成新的文本
- **预训练**：在大量数据上训练，学习语言模式
- **Transformer**：一种神经网络架构，使用注意力机制

### 类比理解：
```
想象你在写作文：
- 你看到了前面的字："今天天气很"
- 你的大脑会预测下一个字：可能是"好"、"冷"、"热"等
- GPT 就是做这件事的 AI 模型
```

## 1.2 项目整体结构

```
mini-gpt/
│
├── 数据层 (Data Layer)
│   └── data/input.txt          # 训练数据（莎士比亚文本）
│
├── 工具层 (Utility Layer)
│   ├── utils.py                # 分词器（文本 ↔ 数字）
│   └── config.py               # 配置参数
│
├── 模型层 (Model Layer)
│   └── model.py                # GPT 架构（核心）
│
├── 训练层 (Training Layer)
│   └── train.py                # 训练脚本
│
└── 应用层 (Application Layer)
    └── generate.py             # 文本生成
```

## 1.3 你需要掌握的知识地图

```
基础知识（必须）
├── Python 编程
│   ├── 类 (Class) 和对象
│   ├── 函数和模块
│   └── 基本数据结构（列表、字典）
│
├── PyTorch 基础
│   ├── Tensor（张量）操作
│   ├── nn.Module（神经网络模块）
│   └── 自动求导（autograd）
│
└── 数学基础
    ├── 矩阵乘法
    ├── 概率（softmax）
    └── 损失函数

进阶知识（重要）
├── 深度学习
│   ├── 神经网络前向传播
│   ├── 反向传播
│   └── 梯度下降
│
└── Transformer 架构
    ├── 注意力机制（Attention）
    ├── 多头注意力
    └── 残差连接
```

---

# 第二部分：数据处理与分词器

## 2.1 为什么需要分词器？

**问题**：计算机不理解文字，只理解数字。

**解决方案**：分词器（Tokenizer）

```python
# 文本 → 数字（编码）
"Hello" → [20, 43, 47, 47, 52]

# 数字 → 文本（解码）
[20, 43, 47, 47, 52] → "Hello"
```

## 2.2 理解 `utils.py` - 字符级分词器

### 第一步：打开并阅读 utils.py

```bash
cat utils.py
```

### 代码详解：

```python
class CharTokenizer:
    """字符级分词器"""
    
    def __init__(self, text):
        # 步骤1：获取所有唯一字符并排序
        chars = sorted(list(set(text)))
        # 例：text = "hello" → chars = ['e', 'h', 'l', 'o']
        
        self.vocab_size = len(chars)
        # vocab_size = 4
        
        # 步骤2：创建字符到数字的映射
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        # stoi = {'e': 0, 'h': 1, 'l': 2, 'o': 3}
        
        # 步骤3：创建数字到字符的映射（反向）
        self.itos = {i: ch for ch, i in self.stoi.items()}
        # itos = {0: 'e', 1: 'h', 2: 'l', 3: 'o'}

    def encode(self, s):
        """文本 → 数字列表"""
        return [self.stoi[ch] for ch in s]
        # "hello" → [1, 0, 2, 2, 3]

    def decode(self, ids):
        """数字列表 → 文本"""
        return "".join([self.itos[i] for i in ids])
        # [1, 0, 2, 2, 3] → "hello"
```

### 动手实践1：测试分词器

创建文件 `test_tokenizer.py`：

```python
from utils import CharTokenizer

# 测试文本
text = "hello world"

# 创建分词器
tokenizer = CharTokenizer(text)

# 查看词汇表
print("字符集:", sorted(list(set(text))))
print("词汇表大小:", tokenizer.vocab_size)
print("\n字符→数字映射 (stoi):")
print(tokenizer.stoi)

# 编码
encoded = tokenizer.encode("hello")
print("\n编码 'hello':", encoded)

# 解码
decoded = tokenizer.decode(encoded)
print("解码回来:", decoded)
```

**运行**：
```bash
python3 test_tokenizer.py
```

### 思考题：
1. 为什么要对字符进行排序？
2. 如果文本中出现了训练时没见过的字符会怎样？
3. 字符级分词 vs 词级分词有什么区别？

---

# 第三部分：Transformer 模型架构

## 3.1 整体架构图

```
输入文本: "Hello"
    ↓
[编码] → [20, 43, 47, 47, 52]
    ↓
[Token Embedding] → 每个数字变成一个向量
    ↓
[Position Embedding] → 加上位置信息
    ↓
[Transformer Block 1]
    ├── Multi-Head Attention（多头注意力）
    ├── Residual Connection（残差连接）
    ├── Layer Norm（层归一化）
    ├── Feed-Forward Network（前馈网络）
    └── Residual Connection
    ↓
[Transformer Block 2]
    ... (重复 6 次)
    ↓
[Transformer Block 6]
    ↓
[Layer Norm]
    ↓
[Linear Layer] → 预测下一个字符的概率
    ↓
输出: 每个可能字符的概率分布
```

## 3.2 核心概念详解

### 3.2.1 什么是 Embedding（嵌入）？

**问题**：数字 `20` 本身没有意义。

**解决**：将每个数字映射到一个高维向量。

```python
# 简化示例
数字 20 → [0.5, -0.3, 0.8, 0.1, ...]  # 384维向量
数字 43 → [0.2, 0.7, -0.5, 0.9, ...]  # 384维向量
```

### 类比：
```
把每个字符想象成一个人
每个人有很多特征（身高、体重、年龄...）
这些特征就是向量的各个维度
```

### 代码位置（model.py）：
```python
self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
# vocab_size: 词汇表大小（65个字符）
# n_embd: 每个字符用多少维表示（384维）
```

### 3.2.2 什么是 Position Embedding（位置嵌入）？

**问题**：模型需要知道字符的顺序。

```
"Hello" 和 "olleH" 应该不同
但如果只看字符本身，模型无法区分顺序
```

**解决**：给每个位置一个独特的向量。

```python
位置 0 → [0.1, 0.2, 0.3, ...]
位置 1 → [0.4, 0.5, 0.6, ...]
位置 2 → [0.7, 0.8, 0.9, ...]
```

### 最终输入：
```
Token Embedding + Position Embedding

"H" 在位置0:
  [0.5, -0.3, 0.8, ...]  (token)
+ [0.1,  0.2, 0.3, ...]  (position)
= [0.6, -0.1, 1.1, ...]  (最终输入)
```

## 3.3 注意力机制（Attention）- 核心中的核心

### 3.3.1 直观理解

**场景**：你在写"今天天气很____"

你的大脑会：
1. **看** 前面的字："今天"、"天气"、"很"
2. **关注** 最重要的信息（"天气"）
3. **预测** 下一个字（"好"、"冷"等）

**注意力机制就是让模型学会"关注"！**

### 3.3.2 注意力的三个步骤

#### 步骤1：计算 Query, Key, Value (Q, K, V)

```python
# 简化理解
Query (查询):  "我想找什么信息？"
Key (键):      "我有什么信息？"
Value (值):    "信息的内容是什么？"
```

#### 实际例子：
```
句子："The cat sat on the mat"
当预测 "mat" 后面的词时：

Query: "mat" 在问："我应该关注什么？"
Key: 每个词说："我是 The/cat/sat/on/the/mat"
Value: 每个词的实际含义向量
```

#### 步骤2：计算注意力分数

```python
# 公式
Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V

# 分解：
1. Q @ K^T: 计算相似度
2. / √d: 缩放（避免数值过大）
3. softmax: 转换成概率（和为1）
4. @ V: 加权求和
```

#### 步骤3：应用 Causal Mask（因果掩码）

**问题**：训练时不能"作弊"（看到未来的词）

```
预测位置 2 的词时，只能看位置 0, 1, 2
不能看位置 3, 4, 5（那是未来）
```

**解决**：使用下三角矩阵

```python
# torch.tril 创建下三角矩阵
[[1, 0, 0, 0],   # 位置0只能看自己
 [1, 1, 0, 0],   # 位置1可以看0,1
 [1, 1, 1, 0],   # 位置2可以看0,1,2
 [1, 1, 1, 1]]   # 位置3可以看0,1,2,3
```

### 3.3.3 代码详解（model.py 的 Head 类）

```python
class Head(nn.Module):
    """单个注意力头"""
    
    def __init__(self, n_embd, head_size, block_size, dropout=0.1):
        super().__init__()
        # 创建 Q, K, V 的线性变换层
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # 注册因果掩码（下三角矩阵）
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x 的形状: (batch, time, channels)
        B, T, C = x.shape
        
        # 计算 Q, K, V
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # 计算注意力分数
        # q @ k^T: (B, T, head_size) @ (B, head_size, T) → (B, T, T)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # 缩放
        
        # 应用因果掩码
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # softmax 转换成概率
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        
        # 加权求和
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, T) @ (B, T, head_size) → (B, T, head_size)
        
        return out
```

### 动手实践2：可视化注意力

创建 `visualize_attention.py`：

```python
import torch
import matplotlib.pyplot as plt

# 模拟注意力权重矩阵
seq_len = 8
attention = torch.tril(torch.ones(seq_len, seq_len))
attention = attention / attention.sum(dim=1, keepdim=True)

# 可视化
plt.figure(figsize=(8, 6))
plt.imshow(attention, cmap='Blues')
plt.colorbar()
plt.title('Causal Attention Pattern')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.savefig('attention_pattern.png')
print("已保存到 attention_pattern.png")
```

## 3.4 多头注意力（Multi-Head Attention）

### 为什么需要多个头？

**类比**：
```
一个人看问题，视角单一
多个人（多个头）看问题，视角全面

头1: 关注语法
头2: 关注语义
头3: 关注上下文
...
```

### 代码（model.py）：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // num_heads  # 每个头的维度
        
        # 创建多个注意力头
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) 
            for _ in range(num_heads)
        ])
        
        # 输出投影
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 每个头独立计算，然后拼接
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
```

## 3.5 前馈网络（Feed-Forward Network）

### 作用：
在注意力之后，进一步处理信息

### 结构：
```
输入 (384维)
  ↓
线性层1 → 1536维 (扩展4倍)
  ↓
ReLU 激活
  ↓
线性层2 → 384维 (压缩回来)
  ↓
输出 (384维)
```

### 代码：

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 扩展
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # 压缩
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
```

## 3.6 Transformer Block（组合起来）

### 结构：
```
输入
  ↓
LayerNorm → MultiHeadAttention → 残差连接
  ↓
LayerNorm → FeedForward → 残差连接
  ↓
输出
```

### 什么是残差连接？

**问题**：深层网络难以训练

**解决**：直接加上原始输入

```python
# 没有残差连接
output = attention(x)

# 有残差连接
output = x + attention(x)
#        ↑   ↑
#        原  新信息
#        始
#        输
#        入
```

### 代码：

```python
class Block(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)
    
    def forward(self, x):
        # 注意力 + 残差
        x = x + self.attn(self.ln1(x))
        
        # 前馈 + 残差
        x = x + self.ffwd(self.ln2(x))
        
        return x
```

---

# 第四部分：训练流程

## 4.1 训练的本质

**目标**：教会模型预测下一个字符

```
输入:  "Hell"
目标:  "o"

输入:  "Hello worl"
目标:  "d"
```

## 4.2 损失函数（Loss Function）

**作用**：衡量模型的预测有多"错"

### Cross-Entropy Loss（交叉熵损失）

```python
# 模型预测（概率分布）
预测: {'a': 0.1, 'b': 0.2, 'c': 0.05, 'd': 0.6, 'e': 0.05}
真实: 'd'

# 好的预测：给正确答案高概率
预测: {'d': 0.9, ...}  # Loss 很小 ✓

# 坏的预测：给错误答案高概率
预测: {'a': 0.9, ...}  # Loss 很大 ✗
```

### 代码（model.py）：

```python
def forward(self, idx, targets=None):
    # ... 前向传播 ...
    
    if targets is None:
        loss = None
    else:
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, targets)
    
    return logits, loss
```

## 4.3 训练循环详解

### 完整流程：

```python
for iteration in range(max_iters):
    # 1. 获取一批训练数据
    x, y = get_batch('train')
    # x: 输入序列 (batch_size, block_size)
    # y: 目标序列 (batch_size, block_size)
    
    # 2. 前向传播：预测
    logits, loss = model(x, y)
    
    # 3. 反向传播：计算梯度
    optimizer.zero_grad()  # 清空上一次的梯度
    loss.backward()        # 计算梯度
    
    # 4. 更新参数
    optimizer.step()
    
    # 5. 定期评估
    if iteration % eval_interval == 0:
        # 在验证集上测试
        val_loss = evaluate(model, val_data)
        print(f"Step {iteration}: train loss {loss:.4f}, val loss {val_loss:.4f}")
```

### 关键函数：`get_batch`

```python
def get_batch(split):
    """生成一批训练数据"""
    data = train_data if split == 'train' else val_data
    
    # 随机选择起始位置
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # 创建输入和目标
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y
```

### 示例：

```
原始数据: "Hello world"
编码: [20, 43, 47, 47, 52, 1, 58, 52, 55, 47, 42]

block_size = 4

批次1:
  x = [20, 43, 47, 47]  → "Hell"
  y = [43, 47, 47, 52]  → "ello"

批次2:
  x = [52, 1, 58, 52]   → "o wo"
  y = [1, 58, 52, 55]   → " wor"
```

## 4.4 优化器（Optimizer）

### AdamW 优化器

**作用**：智能地调整学习速度

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

**参数更新公式（简化）**：
```
新参数 = 旧参数 - 学习率 × 梯度
```

### 学习率（Learning Rate）

```
学习率太大 → 不稳定，跳来跳去
学习率太小 → 训练太慢
```

默认值：`3e-4` (0.0003) 是个不错的起点

---

# 第五部分：文本生成

## 5.1 生成的本质

**自回归生成**：一个字一个字地生成

```
步骤1: 输入 "H"     → 预测 "e"
步骤2: 输入 "He"    → 预测 "l"
步骤3: 输入 "Hel"   → 预测 "l"
步骤4: 输入 "Hell"  → 预测 "o"
...
```

## 5.2 采样策略

### 5.2.1 Temperature（温度）

**控制随机性**

```python
logits = logits / temperature

temperature = 0.1  → 保守（总选最可能的）
temperature = 1.0  → 平衡
temperature = 2.0  → 创新（更随机）
```

### 示例：

```
原始概率: {'a': 0.5, 'b': 0.3, 'c': 0.2}

temperature = 0.5 (更确定):
→ {'a': 0.8, 'b': 0.15, 'c': 0.05}

temperature = 2.0 (更随机):
→ {'a': 0.4, 'b': 0.35, 'c': 0.25}
```

### 5.2.2 Top-K 采样

**只从最可能的 K 个候选中选择**

```python
# top_k = 5
原始: 65个字符都可能被选中
top_k: 只从最可能的5个字符中选

好处：避免选到很不合理的字符
```

### 代码（model.py 的 generate 方法）：

```python
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        # 1. 裁剪上下文（只保留最后 block_size 个）
        idx_cond = idx[:, -self.block_size:]
        
        # 2. 前向传播
        logits, _ = self(idx_cond)
        
        # 3. 只看最后一个位置的预测
        logits = logits[:, -1, :] / temperature
        
        # 4. Top-K 过滤
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # 5. 转换成概率
        probs = F.softmax(logits, dim=-1)
        
        # 6. 采样下一个token
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # 7. 拼接到序列
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
```

---

# 第六部分：实践练习

## 练习1：理解分词器

**任务**：实现一个词级分词器

```python
class WordTokenizer:
    def __init__(self, text):
        # TODO: 按空格分词，创建词汇表
        pass
    
    def encode(self, text):
        # TODO: 文本 → 数字列表
        pass
    
    def decode(self, ids):
        # TODO: 数字列表 → 文本
        pass

# 测试
text = "hello world hello"
tokenizer = WordTokenizer(text)
print(tokenizer.encode("hello"))  # 应该输出 [0] 或 [1]
```

## 练习2：可视化Embedding

```python
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 创建简单的embedding
vocab_size = 10
embed_dim = 50
embedding = nn.Embedding(vocab_size, embed_dim)

# 获取所有embedding向量
all_embeddings = embedding.weight.detach().numpy()

# 降维到2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(all_embeddings)

# 可视化
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
for i in range(vocab_size):
    plt.annotate(str(i), (embeddings_2d[i, 0], embeddings_2d[i, 1]))
plt.title('Token Embeddings (2D)')
plt.savefig('embeddings.png')
```

## 练习3：修改模型参数

**任务**：在 `config.py` 中尝试不同的配置

```python
# 实验1: 小模型（快速测试）
n_embd = 128
n_layer = 3
num_heads = 4
max_iters = 1000

# 实验2: 中等模型（默认）
n_embd = 384
n_layer = 6
num_heads = 6
max_iters = 5000

# 实验3: 大模型（如果GPU足够）
n_embd = 512
n_layer = 8
num_heads = 8
max_iters = 10000
```

**观察**：
- 模型大小对训练时间的影响
- 模型大小对生成质量的影响
- Loss 下降的速度

## 练习4：自定义数据集

**任务**：用自己的文本训练模型

```bash
# 1. 准备你的文本文件
echo "你的中文文本或英文文本" > data/my_text.txt

# 2. 修改 train.py
# 把 'data/input.txt' 改成 'data/my_text.txt'

# 3. 训练
python3 train.py

# 4. 生成
python3 generate.py --prompt "你的提示词"
```

## 练习5：实现学习率调度器

```python
# 在 train.py 中添加
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=max_iters)

for iter in range(max_iters):
    # ... 训练代码 ...
    optimizer.step()
    scheduler.step()  # 更新学习率
    
    if iter % 100 == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Iteration {iter}, LR: {current_lr:.6f}")
```

---

# 学习路线图

## 第1周：基础理解

- [ ] 阅读并理解 utils.py
- [ ] 运行 test_tokenizer.py
- [ ] 理解 Embedding 的概念
- [ ] 可视化注意力矩阵

## 第2周：模型架构

- [ ] 逐行阅读 model.py
- [ ] 理解单头注意力机制
- [ ] 理解多头注意力
- [ ] 理解 Transformer Block

## 第3周：训练与生成

- [ ] 理解训练循环
- [ ] 运行完整训练
- [ ] 实验不同的生成参数
- [ ] 观察 loss 曲线

## 第4周：实验与改进

- [ ] 尝试不同模型大小
- [ ] 使用自己的数据集
- [ ] 实现学习率调度
- [ ] 添加 wandb 日志

---

# 调试技巧

## 1. 打印 Tensor 形状

```python
def forward(self, x):
    print(f"Input shape: {x.shape}")
    x = self.layer1(x)
    print(f"After layer1: {x.shape}")
    # ... 继续
```

## 2. 检查梯度

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
    else:
        print(f"{name}: NO GRADIENT!")
```

## 3. 可视化训练过程

```python
import matplotlib.pyplot as plt

train_losses = []
val_losses = []

for iter in range(max_iters):
    # 训练...
    train_losses.append(train_loss)
    val_losses.append(val_loss)

# 绘制
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.legend()
plt.savefig('loss_curve.png')
```

---

# 常见问题 FAQ

## Q1: 为什么 loss 不下降？

**可能原因**：
- 学习率太大或太小
- 模型太小，容量不够
- 数据有问题
- 梯度爆炸/消失

**解决方法**：
```python
# 1. 调整学习率
learning_rate = 1e-4  # 试试更小的

# 2. 添加梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# 3. 检查数据
print(f"Train data: {train_data[:100]}")
```

## Q2: 生成的文本是乱码？

**原因**：
- 训练不够充分
- Temperature 设置不当

**解决**：
```bash
# 1. 训练更久
max_iters = 10000

# 2. 调整 temperature
python3 generate.py --temperature 0.7  # 更保守
```

## Q3: 内存不够？

**解决**：
```python
# 减小 batch_size
batch_size = 32  # 从 64 减到 32

# 减小 block_size
block_size = 128  # 从 256 减到 128

# 减小模型
n_embd = 256
n_layer = 4
```

---

# 进阶学习资源

## 论文
1. **Attention Is All You Need** (原始 Transformer)
2. **Language Models are Unsupervised Multitask Learners** (GPT-2)

## 教程
1. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
2. [Andrej Karpathy's YouTube: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## 代码
1. [nanoGPT](https://github.com/karpathy/nanoGPT) - 本项目的灵感来源

---

# 总结

## 你已经掌握的技能：

✅ **数据处理**
- 字符级分词
- 文本编码/解码

✅ **模型架构**
- Embedding（Token + Position）
- 自注意力机制
- 多头注意力
- 前馈网络
- Transformer Block

✅ **训练流程**
- 损失函数
- 梯度下降
- 参数更新

✅ **文本生成**
- 自回归生成
- Temperature 采样
- Top-K 采样

## 下一步建议：

1. **实践，实践，再实践！**
   - 在不同数据集上训练
   - 调整超参数
   - 观察模型行为

2. **阅读代码**
   - 每天读一点 model.py
   - 理解每一行的作用

3. **做实验**
   - 改动代码，看看会发生什么
   - 记录你的发现

4. **分享**
   - 写博客记录学习过程
   - 在 GitHub 上分享你的改进

---

**恭喜你！你现在拥有构建自己的语言模型的知识了！🎉**

Keep learning, keep building! 💪
