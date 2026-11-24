# PCVRHyFormer Baseline 架构详解

> 源文件：[code/TAAC_baseline/model.py](../code/TAAC_baseline/model.py)（共 1715 行）
> 任务：腾讯广告大赛 PCVR（Post-Click Conversion Rate）二分类
> 模型类：`PCVRHyFormer`

本文档目标：把 baseline 模型每一处的张量形状、Q/K/V 来源、每一维的物理含义全部展开，作为后续改造的参考底稿。

---

## 0. 符号约定

| 符号 | 含义 | 默认值/来源 |
|---|---|---|
| `B` | batch size | DataLoader |
| `D` | `d_model`，模型主干隐维 | 64 |
| `E` | `emb_dim`，每个 fid 的 embedding 维度 | 64 |
| `H` | `num_heads` | 4 |
| `d_head` | `D / H` | 16 |
| `Nq` | 每条序列对应的 query token 数 (`num_queries`) | 1 |
| `S` | 序列 domain 数 (`num_sequences = len(seq_domains)`)，按 domain 名字典序固定 | 由 schema 决定 |
| `F_d` | domain `d` 的 side-info fid 数 | 由 schema 决定 |
| `L_d` | domain `d` 的最大序列长度 | 由 batch padding 决定 |
| `M` | NS token 总数 `num_ns = num_user_ns + (1 if has_user_dense else 0) + num_item_ns + (1 if has_item_dense else 0)` | 由 NS 配置 |
| `T` | RankMixer 输入 token 数 `T = Nq*S + M` | 派生 |
| `action_num` | 输出 logits 维度（baseline=1） | 1 |

**硬约束**
- `D % H == 0`
- RankMixer `mode='full'`：`D % T == 0`
- RoPE 只作用于 `head_dim` 维度，对 Q/K 旋转，**V 不旋转**
- 所有 `nn.Embedding` 都用 `padding_idx=0`，0 行恒为 0 向量

---

## 1. 顶层架构图

```
┌──────────────────────── ModelInput (NamedTuple) ─────────────────────────┐
│ user_int_feats    : LongTensor  (B, total_user_int_dim)                  │
│ item_int_feats    : LongTensor  (B, total_item_int_dim)                  │
│ user_dense_feats  : FloatTensor (B, user_dense_dim)                      │
│ item_dense_feats  : FloatTensor (B, item_dense_dim)   # baseline=0       │
│ seq_data          : { domain : LongTensor (B, F_d, L_d) }                │
│ seq_lens          : { domain : LongTensor (B,) }                         │
│ seq_time_buckets  : { domain : LongTensor (B, L_d) }                     │
└──────────────────────────────────────────────────────────────────────────┘
            │                                              │
            ▼                                              ▼
   ┌─────────────────────┐                       ┌─────────────────────┐
   │  NS Tokenizer       │                       │  Seq Embedding      │
   │  (User/Item)        │                       │  (per domain)       │
   │  + Dense Proj       │                       │  + Time Bucket Emb  │
   │                     │                       │                     │
   │  → ns_tokens        │                       │  → seq_tokens_list  │
   │      (B, M, D)      │                       │  S × (B, L_d, D)    │
   │                     │                       │  + masks (B, L_d)   │
   └─────────────────────┘                       └─────────────────────┘
            │                                              │
            └─────────────────┬────────────────────────────┘
                              ▼
                ┌──────────────────────────────┐
                │ MultiSeqQueryGenerator       │
                │  per seq i:                  │
                │   global_info = LN([ns_flat ;│
                │                  meanpool_i])│
                │   q_i = stack(FFN_{i,1..Nq}( │
                │            global_info))     │
                │  → q_tokens_list             │
                │     S × (B, Nq, D)           │
                └──────────────────────────────┘
                              │
                              ▼  (+ ns_tokens, + seq_tokens_list, + masks)
              ┌────────────────────────────────────────┐
              │  N × MultiSeqHyFormerBlock             │
              │   ┌──────────────────────────────────┐ │
              │   │ (a) Seq Encoder × S              │ │
              │   │     swiglu | transformer | longer│ │
              │   │ (b) CrossAttention × S           │ │
              │   │     Q ← q_i, K=V ← evolved_seq_i │ │
              │   │ (c) Concat([dec_q_0..S-1, ns])   │ │
              │   │     → (B, T, D)                  │ │
              │   │ (d) RankMixerBlock               │ │
              │   │     token-mix + per-token FFN    │ │
              │   │ (e) Split → q_list, ns           │ │
              │   └──────────────────────────────────┘ │
              └────────────────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │ Output Projection            │
                │  cat(q_list, dim=1)          │
                │   (B, Nq*S, D)               │
                │  view (B, Nq*S*D)            │
                │  Linear(Nq*S*D → D) + LN     │
                │  → (B, D)                    │
                └──────────────────────────────┘
                              │
                              ▼
                ┌──────────────────────────────┐
                │ Classifier                   │
                │  Linear D→D  + LN + SiLU +   │
                │  Dropout + Linear D→action_num│
                │  → logits (B, action_num)    │
                └──────────────────────────────┘
```

---

## 2. ModelInput 字段维度明细

[`model.py:11-18`](../code/TAAC_baseline/model.py#L11-L18)

| 字段 | dtype | 形状 | 每一维含义 |
|---|---|---|---|
| `user_int_feats` | int64 | `(B, total_user_int_dim)` | 全部 user 整数特征沿列拼接：每个 fid 占 `length` 列（单值=1，多值>1，padding=0）；列布局由 `feature_specs=[(vocab_size, offset, length), ...]` 描述 |
| `item_int_feats` | int64 | `(B, total_item_int_dim)` | 同上，item 侧 |
| `user_dense_feats` | float32 | `(B, user_dense_dim)` | user 稠密特征向量 |
| `item_dense_feats` | float32 | `(B, item_dense_dim)` | baseline 中 `item_dense_dim=0`，是空 tensor |
| `seq_data[domain]` | int64 | `(B, F_d, L_d)` | 第 1 维：batch；第 2 维：该 domain 的 `F_d` 个 side-info fid（含主 id）；第 3 维：时间步（最近行为靠右） |
| `seq_lens[domain]` | int64 | `(B,)` | 每个样本该 domain 的有效长度 ∈ `[0, L_d]` |
| `seq_time_buckets[domain]` | int64 | `(B, L_d)` | 相对时间桶 id ∈ `[0, num_time_buckets)`；padding 位为 0 → 通过 `Embedding(padding_idx=0)` 得到零向量 |

注意：`seq_data` 是 `(B, F_d, L_d)` 而**非** `(B, L_d, F_d)`——见 [`_embed_seq_domain` 中的 `seq[:, i, :]`](../code/TAAC_baseline/model.py#L1563)。

---

## 3. NS Tokenizer

NS = "Non-Sequence"，即把用户/物品的静态特征（无时间轴）压成 `M` 个 D 维 token，供后续 attention 处理。两种实现共存，由 `ns_tokenizer_type` 选择。

### 前置概念：feature_specs 与 int_feats 的列布局

每个 fid 对应一条 `(vocab_size, offset, length)` 描述：
- `offset`：该 fid 在 `int_feats` 大向量里从第几列开始
- `length`：占几列（单值特征=1，多值特征=允许的最大取值个数）
- 值=0 表示 padding（没有该取值），所有 Embedding 都用 `padding_idx=0`

**举例**：用户有 3 个 fid——性别（单值）、年龄段（单值）、兴趣标签（多值，最多 3 个）：

```
int_feats 列布局：[性别id | 年龄段id | 兴趣标签1 | 兴趣标签2 | 兴趣标签3]
一行示例：        [  2    |    5    |    123   |    456   |     0    ]
                                                              ↑ 0 = 没有第3个标签
feature_specs = [(3, 0, 1), (10, 1, 1), (5000, 2, 3)]
```

---

### 3.1 GroupNSTokenizer  [`model.py:988-1067`](../code/TAAC_baseline/model.py#L988-L1067)

人工把 fid 划分成若干 group，**每个 group 对应一个 NS token**，token 数 = group 数。

**每个 group 的处理流程（以上面例子为例，把 fid 0+1 分一组，fid 2 单独一组）：**

**第一组（性别 + 年龄段）：**
- fid 0（单值）：`Embedding(4, E)[int_feats[:, 0]]` → `(B, E)`
- fid 1（单值）：`Embedding(11, E)[int_feats[:, 1]]` → `(B, E)`
- 两个 embedding 沿最后一维拼接 → `(B, 2E)`
- `Linear(2E → D)` + `LN` + `SiLU` + `unsqueeze(1)` → `(B, 1, D)`，这是第一个 NS token

**第二组（兴趣标签，多值）：**
- fid 2（多值）：`Embedding(5001, E)[int_feats[:, 2:5]]` → `(B, 3, E)`
- 做 masked mean：`mask = (vals != 0)`，对有效标签取平均 → `(B, E)`
  - 例如 `[123, 456, 0]`：只对前两行求平均，忽略 padding 的 0
- `Linear(E → D)` + `LN` + `SiLU` + `unsqueeze(1)` → `(B, 1, D)`，这是第二个 NS token

**最终：** `cat([token_0, token_1], dim=1)` → `(B, 2, D)`

输出 `(B, num_groups, D)`，其中 `num_groups = len(ns_groups)`。

---

### 3.2 RankMixerNSTokenizer  [`model.py:1070-1189`](../code/TAAC_baseline/model.py#L1070-L1189)

**baseline 默认使用**。token 数量由超参 `num_ns_tokens` 自由设定，不受 group 数量约束。

核心思路：把所有 fid 的 embedding 拼成一个大向量，然后等长切块，每块各自过独立 FFN 生成一个 token。

**以同样的 3 个 fid 为例，想输出 4 个 NS token：**

**第一步：所有 fid 依次 embedding（方式同 GroupNS），全部拼成一个大向量**
- fid 0 → `(B, E)`，fid 1 → `(B, E)`，fid 2（多值 masked mean）→ `(B, E)`
- `cat` → `(B, 3E)`，例如 E=64 则为 `(B, 192)`

**第二步：右 pad 让总长整除 num_ns_tokens**
- `192 / 4 = 48` 整除，无需 pad，仍为 `(B, 192)`

**第三步：等长切成 4 段，每段过独立 FFN**
- 切成 4 段，各 `(B, 48)`
- 每段各自过 `Linear(48 → D)` + `LN` + `SiLU` → `(B, D)`
  - 注意：4 个 FFN 参数**完全独立**（`nn.ModuleList` 里各自一套权重）
  - 每个 FFN 自己学"从这段 embedding 里提取什么信息"

**第四步：** `cat` → `(B, 4, D)`

**设计意图**：`num_ns_tokens` 控制 NS 侧的表达粒度，切分虽然是盲切（不按语义），但后续 attention 可以对不同 token 差异化关注，FFN 也能学到一定的语义分布，同时避免了人工设计分组的麻烦。

---

### 3.3 Dense Token  [`model.py:1300-1312`](../code/TAAC_baseline/model.py#L1300-L1312)

Dense 特征（float，如点击率、在线时长）直接线性投影，**不需要 Embedding 查表**：

```
user_dense_feats (B, user_dense_dim)
  → Linear(user_dense_dim → D) + LN + SiLU
  → unsqueeze(1)
  → (B, 1, D)
```

baseline 中 `item_dense_dim=0`，item 侧没有 dense token。

---

### 3.4 ns_tokens 最终拼接

[`model.py:1640-1649`](../code/TAAC_baseline/model.py#L1640-L1649)：

```
ns_tokens = cat([
    user_ns,          # (B, num_user_ns, D)   ← user int 特征
    user_dense_tok?,  # (B, 1, D)             ← user dense（若有）
    item_ns,          # (B, num_item_ns, D)   ← item int 特征
    item_dense_tok?,  # (B, 1, D)             ← item dense（若有）
], dim=1)             # → (B, M, D)
```

`M = num_user_ns + (1 if user_dense) + num_item_ns + (1 if item_dense)`，是整个模型里 NS 侧的 token 总数。

---

## 4. Sequence Embedding

`PCVRHyFormer._embed_seq_domain` [`model.py:1544-1574`](../code/TAAC_baseline/model.py#L1544-L1574)。每个 domain 独立调用，把行为序列的每个时间步变成一个 D 维 token。

### 输入数据结构

以"点击序列"domain 为例，假设有 3 个 side-info fid（广告id、广告类别、广告位置）：

```
seq_data["click"]      形状 (B, 3, L)
                              ↑  ↑
                           3个fid  L个时间步（最近行为靠右，左边padding=0）

seq_time_buckets["click"]  形状 (B, L)   # 每个时间步距当前的相对时间桶 id（0=padding）
seq_lens["click"]          形状 (B,)     # 每个样本的实际有效长度
```

注意：`seq_data` 的第 2 维是 fid，第 3 维是时间步，与 NS 的 `int_feats` 布局不同（`seq[:, i, :]` 取第 i 个 fid 的完整时间序列）。

### 逐步处理

**第一步：每个 fid 独立 Embedding，得到 (B, L, E)**

```
seq[:, 0, :]  (B, L) → Embedding(vs_0+1, E, padding_idx=0) → (B, L, E)   # 广告id
seq[:, 1, :]  (B, L) → Embedding(vs_1+1, E, padding_idx=0) → (B, L, E)   # 广告类别
seq[:, 2, :]  (B, L) → Embedding(vs_2+1, E, padding_idx=0) → (B, L, E)   # 广告位置
```

- padding 位（值=0）通过 `padding_idx=0` 自动得到零向量
- 若 `vocab_size > seq_id_threshold`（默认 10000）且处于 training，额外施加 `Dropout(2 * dropout_rate)`，防止高基数 ID 特征过拟合

**第二步：沿最后一维 cat，得到 (B, L, F×E)**

```
cat([fid0_emb, fid1_emb, fid2_emb], dim=-1) → (B, L, 3E)
```

每个时间步现在是一个 `F×E` 维向量，包含该时刻所有 side-info 的拼接。

**第三步：线性投影压缩到 D，得到 (B, L, D)**

```
Linear(F×E → D) + LN + GeLU → (B, L, D)
```

每个 domain 有自己独立的 proj 参数。

**第四步：加时间桶 Embedding**

```
time_embedding(seq_time_buckets)   → (B, L, D)
token_emb = token_emb + time_emb   → (B, L, D)
```

- `time_embedding = Embedding(num_time_buckets, D, padding_idx=0)`
- 时间桶编码"这个行为距今多久"，直接加到 token 上（类似位置编码的加法形式）
- padding 位的时间桶 id=0，通过 `padding_idx=0` 产生零向量，加上去不影响 padding token

**最终输出：**

```
seq_tokens_list[i]  : (B, L_d, D)    # 每个时间步一个 token，融合了该时刻所有 side-info + 相对时间
key_padding_mask[i] : (B, L_d)       # True = padding，由 seq_lens 生成（idx >= seq_len 的位置为 True）
```

### 与 NS Tokenizer 的对比

| | NS Tokenizer | Seq Embedding |
|---|---|---|
| 输入 | 静态特征，无时间轴 | 行为序列，有时间轴 |
| 每样本 token 数 | 固定 M 个 | L 个（含 padding）|
| 时间信息 | 无 | 有，时间桶加到每个 token 上 |
| 多值特征处理 | masked mean pool 成一个向量 | 无需，每个时间步独立处理 |
| 后续角色 | 进入 RankMixer 做全局特征混合 | 进入 SeqEncoder 做序列演化，再被 Q-token cross-attend |

---

## 5. MultiSeqQueryGenerator

[`model.py:415-494`](../code/TAAC_baseline/model.py#L415-L494)。为每条 sequence 独立生成 `Nq` 个 query token。

```
ns_tokens         (B, M, D)
seq_tokens_list   S × (B, L_i, D)
seq_padding_masks S × (B, L_i)
    │
    │ ns_flat = ns_tokens.view(B, M*D)      # (B, M*D)
    │
    │ for i in 0..S-1:
    │   valid_mask = ~seq_padding_masks[i]                      # (B, L_i)
    │   seq_pooled_i = sum(seq_i * valid) / count               # (B, D)
    │   global_info  = cat([ns_flat, seq_pooled_i], dim=-1)     # (B, (M+1)*D)
    │   global_info  = LayerNorm(global_info)                   # 防梯度爆炸
    │   for j in 0..Nq-1:
    │     q_{i,j} = FFN_{i,j}(global_info)                      # (B, D)
    │     # FFN_{i,j} = Linear((M+1)*D → 4D) → SiLU → Linear(4D → D) → LN
    │   q_tokens_i = stack(q_{i,*}, dim=1)                      # (B, Nq, D)
    ▼
q_tokens_list : S × (B, Nq, D)
```

注意：`MultiSeqQueryGenerator` 的 FFN 是 `S * Nq` 个**完全独立**的 FFN（按 `[seq_idx][q_idx]` 索引），不共享参数。

---

## 6. Rotary Position Embedding (RoPE)

启用条件：`use_rope=True` → 创建 `self.rotary_emb = RotaryEmbedding(dim=d_head, base=rope_base)`。

### 6.1 RotaryEmbedding  [`model.py:26-64`](../code/TAAC_baseline/model.py#L26-L64)

- `inv_freq = 1 / base^(arange(0,d_head,2)/d_head)`，形状 `(d_head/2,)`
- 对位置 `t` 计算 `freqs = outer(t, inv_freq)` → `(L, d_head/2)`
- `emb = cat([freqs, freqs], -1)` → `(L, d_head)`
- 缓存 `cos_cached / sin_cached`，形状 `(1, max_seq_len, d_head)`
- `forward(seq_len, device)` 返回 `cos[:, :seq_len, :], sin[:, :seq_len, :]`

### 6.2 apply_rope_to_tensor  [`model.py:74-92`](../code/TAAC_baseline/model.py#L74-L92)

输入 `x (B, H, L, d_head)`，按 `x * cos + rotate_half(x) * sin` 旋转后半通道。

---

## 7. RoPEMultiheadAttention（核心 Q/K/V 块）

[`model.py:117-241`](../code/TAAC_baseline/model.py#L117-L241)。所有 attention（self / cross / longer）都通过它。

### 7.1 数据流

```
query (B, Lq, D)  key (B, Lk, D)  value (B, Lk, D)
    │
    │ Q = W_q(query)   (B, Lq, D) ─── reshape ──→ (B, H, Lq, d_head)
    │ K = W_k(key)     (B, Lk, D) ─── reshape ──→ (B, H, Lk, d_head)
    │ V = W_v(value)   (B, Lk, D) ─── reshape ──→ (B, H, Lk, d_head)
    │
    │ if rope_cos / rope_sin given:
    │   K = apply_rope(K, rope_cos, rope_sin)            # 始终对 K 旋转
    │   if rope_on_q:
    │     q_cos = q_rope_cos or rope_cos
    │     q_sin = q_rope_sin or rope_sin
    │     Q = apply_rope(Q, q_cos, q_sin)                # Q 可独立位置
    │
    │ mask 合成（bool, True=attend）：
    │   key_padding_mask (B, Lk)  → ~kpm.unsqueeze(1).unsqueeze(2)
    │                              → (B, 1, 1, Lk) → expand (B, H, Lq, Lk)
    │   attn_mask (Lq, Lk) (additive, -inf=不可见)
    │                              → (attn_mask==0) bool  → expand 同上
    │   两者 AND
    │
    │ out = F.scaled_dot_product_attention(Q, K, V, attn_mask=..., dropout_p=...)
    │       → (B, H, Lq, d_head)
    │ out = nan_to_num(out, 0.0)                # all-padding 行的 NaN→0（残差侧保留输入）
    │ out = transpose+contiguous+view → (B, Lq, D)
    │
    │ Gate: G = W_g(query)  (B, Lq, D)
    │       out = out * sigmoid(G)
    │   # W_g 初始化：weight=0, bias=1 → sigmoid(1)≈0.731；
    │   # 一开始几乎"恒等放大"，训练后 Gate 能学到 token-wise 抑制
    │
    │ out = W_o(out)         (B, Lq, D)
    ▼
返回 (out, None)
```

### 7.2 关键点

- `rope_on_q` 由调用方决定：
  - `TransformerEncoder`/`LongerEncoder` 的 self-attn → `rope_on_q=True`（Q 与 K 都需要序列位置）
  - `CrossAttention`（query=Nq 个抽象 query token）→ `rope_on_q=False`（query 不在序列时间轴上）
- `q_rope_cos/sin` 用于 LongerEncoder 第一层 cross-attn：query 是从全序列 gather 出的 top-k token，它们在原序列里的位置可能不连续，因此从全局 cos/sin 缓存里按 `position_indices` 单独 gather 出对应位置的旋转量喂给 Q，而 K 仍用整段 `rope_cos/sin`。
- V 永远不旋转（标准 RoPE 约定）。

---

## 8. 三种 Sequence Encoder

由 `create_sequence_encoder(encoder_type, ...)` [`model.py:811-842`](../code/TAAC_baseline/model.py#L811-L842) 选择。

### 8.1 SwiGLUEncoder  [`model.py:502-541`](../code/TAAC_baseline/model.py#L502-L541)

无注意力，纯 FFN：
```
x (B, L, D) ─→ residual
            ─→ LN ─→ SwiGLU(D, hidden_mult)
                       │  fc:    Linear(D → 2*4D)
                       │  split → x1, x2  (B, L, 4D)
                       │  out = x1 * SiLU(x2)
                       │  fc_out: Linear(4D → D)
                       └─→ Dropout
            ─→ + residual
output (B, L, D)
```
mask 透传，不参与计算。

### 8.2 TransformerEncoder  [`model.py:544-614`](../code/TAAC_baseline/model.py#L544-L614)

标准 Pre-LN Encoder Layer（带 RoPE）：
```
x (B, L, D)
  ├─ LN → RoPEMultiheadAttention(Q=K=V=x, rope_on_q=True, key_padding_mask)
  │      → (B, L, D)
  └─ + residual

x ├─ LN → Linear(D→4D) → GELU → Dropout → Linear(4D→D) → Dropout
  └─ + residual
output (B, L, D)
```

### 8.3 LongerEncoder（top-k 压缩）  [`model.py:616-808`](../code/TAAC_baseline/model.py#L616-L808)

**自适应**：根据当前 L 与 `top_k` 的关系切换 cross/self 模式。

#### 第一次调用（L > top_k）：cross-attention 模式

```
x (B, L, D)        key_padding_mask (B, L)   rope_cos/sin (1, L, d_head)
    │
    │ q, new_mask, pos_indices = _gather_top_k(x, key_padding_mask)
    │   # q          : (B, top_k, D)        每个样本最近 top_k 个有效 token；不足处零填充
    │   # new_mask   : (B, top_k)           True=padding（前 top_k - actual_k 个）
    │   # pos_indices: (B, top_k)           原序列里的绝对位置 idx
    │
    │ Pre-LN: q_normed = norm_q(q),  kv_normed = norm_kv(x)
    │
    │ q_rope_cos = gather(rope_cos.expand(B,-1,-1), 1, pos_indices.unsqueeze(-1).expand(-1,-1,d_head))
    │   # (B, top_k, d_head)        top_k 个 token 各自的绝对位置 RoPE
    │ q_rope_sin = gather(rope_sin, ...)        # 同上
    │
    │ attn_out = RoPEMultiheadAttention(
    │     query=q_normed, key=kv_normed, value=kv_normed,
    │     key_padding_mask=key_padding_mask,    # 注意：用原始 (B, L) 的 mask
    │     rope_cos=rope_cos, rope_sin=rope_sin, # K 用整段 RoPE
    │     q_rope_cos=q_rope_cos, q_rope_sin=q_rope_sin,
    │ )                                          # → (B, top_k, D)
    │ out = q + attn_out
    ▼
```

第一次 cross-attn 不施加 causal mask（Q/K 长度不同）。

#### 后续调用（L ≤ top_k）：self-attention 模式

```
x (B, L, D)，L ≤ top_k
    │
    │ Pre-LN: x_normed = norm_q(x)               # 共享 norm_q
    │ attn_mask = causal triangular (L,L) if self.causal else None
    │ attn_out = RoPEMultiheadAttention(Q=K=V=x_normed, kpm, attn_mask, rope_cos, rope_sin)
    │ out = x + attn_out
    ▼
```

#### 公共 FFN 尾部
```
out ─→ + residual
    ─→ ffn_norm → Linear(D→4D) → GELU → Dropout → Linear(4D→D) → Dropout
    ─→ + residual
返回 (out, new_mask)
```

最终 LongerEncoder 第一次调用把 `(B, L, D)` 压成 `(B, top_k, D)`，后续层固定在 `(B, top_k, D)` 上做 self-attn。

---

## 9. CrossAttention

[`model.py:244-312`](../code/TAAC_baseline/model.py#L244-L312)。Q ← Q-token，K=V ← evolved sequence。

```
query     (B, Nq, D)
key_value (B, L,  D)
key_padding_mask (B, L)
rope_cos/sin (1, L, d_head)
    │
    │ residual = query
    │ query     = norm_q(query)
    │ key_value = norm_kv(key_value)
    │
    │ out = RoPEMultiheadAttention(
    │     query=query, key=key_value, value=key_value,
    │     key_padding_mask=key_padding_mask,
    │     rope_cos=rope_cos, rope_sin=rope_sin,
    │     # rope_on_q=False（构造时设定），Q 不旋转，K 旋转
    │ )                                  # (B, Nq, D)
    │ out = residual + out
    ▼
返回 (B, Nq, D)
```

设计意图：query 是抽象的"信息槽"，没有时间坐标；K/V 是序列 token，需要带位置感。所以 RoPE 只施加在 K 上。

---

## 10. RankMixerBlock（Query Boosting）

[`model.py:315-412`](../code/TAAC_baseline/model.py#L315-L412)。三种 mode：

### mode='full'（默认）：参数自由的 token 混合 + 共享 FFN

```
Q (B, T, D)                     T = Nq*S + M ；要求 D % T == 0；d_sub = D / T
    │
    │ token_mixing(Q):
    │   view (B, T, T, d_sub)            # 把 D 切成 T 个子空间
    │   transpose(1, 2).contiguous()     # 交换 (token 轴 ↔ 子空间轴)
    │   view (B, T, D)
    │ → Q_hat (B, T, D)                  # 完全无参数的张量重排（每个 token 携带其它所有 token 在对应子空间的信息）
    │
    │ x = LN(Q_hat)
    │ x = Linear(D → 4D)(x)
    │ x = GELU(x)
    │ x = Dropout(x)
    │ Q_e = Linear(4D → D)(x)            # 所有 token 共享 FFN 参数
    │
    │ Q_boost = Q + Q_e                  # 残差从原 Q（不是 Q_hat）拉
    │ Q_boost = post_norm(Q_boost)
    ▼
(B, T, D)
```

### mode='ffn_only'：跳过 mixing，只过 FFN+残差

```
Q_hat = Q
之后流程同上
```

### mode='none'：恒等映射

直接 `return Q`，不创建任何子模块。

---

## 11. MultiSeqHyFormerBlock（一个 HyFormer 层）

[`model.py:850-980`](../code/TAAC_baseline/model.py#L850-L980)。模型堆叠 `num_hyformer_blocks` 层这种 block。

```
输入：
  q_tokens_list      S × (B, Nq, D)
  ns_tokens          (B, M, D)
  seq_tokens_list    S × (B, L_i, D)
  seq_padding_masks  S × (B, L_i)
  rope_cos_list/sin_list   S × (1, L_i, d_head) 或 None

(1) Sequence Evolution（独立 per seq）
    for i in 0..S-1:
      next_seq_i, mask_i = seq_encoders[i](
          seq_tokens_list[i], seq_padding_masks[i],
          rope_cos=rope_cos_list[i], rope_sin=rope_sin_list[i],
      )
      # SwiGLU/Transformer 输出 (B, L_i, D)、mask 不变
      # LongerEncoder  第一次 (B, L_i, D) → (B, top_k, D)，mask → (B, top_k)；之后保持 (B, top_k, D)

(2) Query Decoding（独立 per seq）
    for i in 0..S-1:
      decoded_q_i = cross_attns[i](
          q_tokens_list[i], next_seq_i, mask_i,
          rope_cos=rope_cos_list[i], rope_sin=rope_sin_list[i],
      )                                   # (B, Nq, D)

(3) Token Fusion
    combined = cat([decoded_q_0, ..., decoded_q_{S-1}, ns_tokens], dim=1)
                                          # (B, Nq*S + M, D) = (B, T, D)

(4) Query Boosting
    boosted = mixer(combined)             # (B, T, D)

(5) Split
    next_q_list = [boosted[:, i*Nq:(i+1)*Nq, :]  for i in 0..S-1]  # S × (B, Nq, D)
    next_ns     = boosted[:, Nq*S:, :]                              # (B, M, D)

返回 (next_q_list, next_ns, next_seqs, next_masks)
```

注意 `next_seqs` 和 `next_masks` 透传给下一层（这是 LongerEncoder 在第二层及之后只需 self-attn 的原因——上一层已经把它压到 `top_k` 长度）。

---

## 12. 主前向 PCVRHyFormer.forward

[`model.py:1634-1675`](../code/TAAC_baseline/model.py#L1634-L1675)。把上面的拼起来：

```
inputs : ModelInput

① NS Tokenize
    user_ns = user_ns_tokenizer(user_int_feats)        # (B, num_user_ns, D)
    item_ns = item_ns_tokenizer(item_int_feats)        # (B, num_item_ns, D)
    可选 user_dense_tok / item_dense_tok                # (B, 1, D) 各
    ns_tokens = cat([...], dim=1)                      # (B, M, D)

② Seq Embed (per domain)
    for domain in seq_domains (字典序固定):
      seq_tokens_list[i] = _embed_seq_domain(...)      # (B, L_d, D)
      seq_masks_list[i]  = _make_padding_mask(seq_lens[domain], L_d)   # (B, L_d)

③ Query Generation
    q_tokens_list = MultiSeqQueryGenerator(ns_tokens, seq_tokens_list, seq_masks_list)
                                                       # S × (B, Nq, D)

④ Embedding Dropout（仅 training）+ N × MultiSeqHyFormerBlock
    for block in self.blocks:
      若启用 RoPE：对每个 seq 计算 (cos, sin)（基于当前 L_d，可能是 top_k）
      block(...) 输出更新后的 q_list / ns / seqs / masks

⑤ Output Projection
    all_q  = cat(curr_qs, dim=1)             # (B, Nq*S, D)
    output = all_q.view(B, Nq*S*D)
    output = output_proj(output)             # Linear(Nq*S*D → D) + LN → (B, D)
    # 注意：NS token 和最终的 seq tokens 都不进入输出层；只有所有 Q tokens

⑥ Classifier
    logits = clsfier(output)
           = Linear(D→D) → LN → SiLU → Dropout → Linear(D→action_num)
           # → (B, action_num)
```

`predict()` 与 `forward()` 一致，但强制关闭 dropout，并且额外返回 `output (B, D)` 作为表示向量。

---

## 13. 子图 A：MultiSeqHyFormerBlock 内部

```
        q_tokens_list[0]  ─┐
        q_tokens_list[1]  ─┤   (各自 (B, Nq, D))
                ...        │
        q_tokens_list[S-1]─┤
                           │
seq_tokens_list[0] ──► seq_encoder_0 ──► next_seq_0 ──┐
seq_tokens_list[1] ──► seq_encoder_1 ──► next_seq_1 ──┤
        ...                                            │
seq_tokens_list[S-1]──► seq_encoder_{S-1} ──► next_seq_{S-1}─┐
                           │                                │
                           ▼                                ▼
   for each i:   cross_attn_i(Q=q_i, K=V=next_seq_i, mask_i, rope_i)
                       ─► decoded_q_i (B, Nq, D)
                                   │
                                   ▼
                cat([decoded_q_0..S-1, ns_tokens], dim=1)
                          (B, Nq*S + M, D) = (B, T, D)
                                   │
                                   ▼
                          RankMixerBlock
                          (B, T, D)
                                   │
                                   ▼
       split → next_q_list (S × (B, Nq, D)), next_ns (B, M, D)
```

## 子图 B：RoPEMultiheadAttention 的 Q/K/V 路径

```
query (B, Lq, D)     key (B, Lk, D)     value (B, Lk, D)
   │                   │                   │
  W_q                 W_k                 W_v          (Linear D→D)
   │                   │                   │
view+transpose       view+transpose      view+transpose
   ▼                   ▼                   ▼
Q (B,H,Lq,d_head)  K (B,H,Lk,d_head)   V (B,H,Lk,d_head)
   │                   │                   │
  RoPE if rope_on_q   RoPE always         (no RoPE)
   │                   │                   │
   └──────┬────────────┴──────┬────────────┘
          │                    │
          ▼                    ▼
  scaled_dot_product_attention(Q, K, V, attn_mask, dropout_p)
          │
          ▼  (B,H,Lq,d_head)
   nan_to_num + transpose + view
          │
          ▼  (B, Lq, D)
       sigmoid(W_g(query)) ⊙ ·     # token-wise gate
          │
          ▼
        W_o(·)  →  (B, Lq, D)
```

---

## 14. 关键超参与维度约束速查

| 约束 | 出处 |
|---|---|
| `D % H == 0` | `RoPEMultiheadAttention.__init__` |
| `D % T == 0`，其中 `T = Nq*S + M` | `RankMixerBlock(mode='full')` 与 `PCVRHyFormer.__init__` 校验 |
| `output_proj` 输入维 = `Nq * S * D`（NS 不进入输出） | `_run_multi_seq_blocks` 末尾 |
| 所有 `nn.Embedding` 都用 `padding_idx=0`，0 行恒为 0 | `_init_params` 强制置零 |
| RoPE 仅旋转 Q/K，不旋转 V | `RoPEMultiheadAttention.forward` |
| `seq_data` 形状是 `(B, F_d, L_d)` | `_embed_seq_domain` 内 `seq[:, i, :]` |
| LongerEncoder：第一层做 cross 把 L→top_k；后续层 self-attn 在 top_k 上 | `LongerEncoder.forward` 的 if/else |
| `seq_id_emb_dropout` 概率 = `2 * dropout_rate`，仅作用于 ID 列且仅在 training | `_embed_seq_domain` |

---

## 15. 不在本文档范围

- 训练循环 / 损失（BCE 或 focal） / 双优化器（Adagrad+AdamW）/ AMP / EMA → 见 `trainer.py`
- Parquet 数据加载、buffer、shuffle、time-bucket 计算 → 见 `dataset.py` 与 [TAAC_baseline/README.md](../code/TAAC_baseline/README.md)
- NS group 的具体语义划分 → 见 `ns_groups.json`
