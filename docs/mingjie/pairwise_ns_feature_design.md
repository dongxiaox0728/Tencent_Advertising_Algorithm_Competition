# Int-Dense Pairwise NS Feature 设计方案

## 问题描述

数据中有 8 个 fid（62-66, 89-91），每个 fid 是一组**元素级对齐的 (int_id, dense_stat) 对**：

```
user_int_feats_62:   [id₁,   id₂,   id₃  ]   # 离散实体 ID
user_dense_feats_62: [10.5,  20.0,  15.5 ]   # 每个 ID 对应的连续统计值
```

两个列表**等长、逐元素对应**：第 i 个 ID 和第 i 个统计值描述的是同一个实体。统计值的含义可能是：停留时长、点击率、转化得分、交互次数等（具体语义未公开，但都是"越大越重要"的正向指标）。

**核心建模问题**：给定一组 `(id, stat)` 对，如何把它们聚合成一个 D 维向量（NS token），让模型能够利用"哪些实体重要、哪些次要"这类信息？

---

## 现有方案（V6/V9 参考基线）

V6 做的是 **residual additive MLP**（62-66）和 **quantile embedding**（89-91）：

```python
# V6（62-66）
norm_stat = sign(x) * log1p(|x|)               # 归一化统计值
pair_emb  = id_emb + MLP([id_emb, norm_stat])  # 加性残差
repr      = masked_mean_pool(pair_emb)          # 均值池化
```

这是一个合理的起点，但有几个值得改进的地方：

1. **聚合阶段忽略了 stat 的重要性差异**：均值池化对所有 id 等权相加，stat=1 和 stat=100 的实体贡献完全相同，白白丢掉了"谁更重要"这个信息。

2. **MLP 的交互方式是加性的**：`id_emb + MLP(...)` 的结构意味着 stat 只能整体地给 id_emb 加一个修正量，无法对 id_emb 的不同维度做差异化的放大或压缩。

3. **各实体之间没有交互**：每个 (id, stat) 对是独立编码的，entity₁ 和 entity₂ 的 embedding 互相不知道对方的存在，无法捕捉"用户同时对 A 和 B 都有偏好"这类组合信号。

---

## 方案一：统计值加权池化（Stat-Weighted Pooling）

### 原理

最简单的改进：把均值池化换成**加权池化**，用 `dense_stat` 的大小来决定每个实体在聚合时的权重。stat 越大的实体，在最终表示里占的比重越大。

```python
norm_stat = sign(x) * log1p(|x|)       # (L,)  对每个 stat 做对数归一化
w = softmax(norm_stat / temperature)    # (L,)  把归一化 stat 转换为权重
repr = sum(w_i * id_emb_i, dim=0)      # (D,)  对 id_emb 做加权求和
```

**各量说明：**

- `x`：原始统计值，如 `[10.5, 20.0, 15.5]`。
- `norm_stat`：对 `x` 做 `sign(x) * log1p(|x|)` 变换后的结果。这一步是为了压缩量级——原始 stat 可能跨度很大（比如 1 到 10000），直接用会让大值主导一切；对数变换后量级更均匀，小值不会被完全忽视。`sign(x)` 保留符号，支持负值的 stat。
- `L`：这个 fid 里的实体数量（列表长度），如 `L=3`。
- `temperature`：一个正数超参数（可以设成可学习的标量），控制权重的"集中程度"。temperature 越小，softmax 越尖锐（权重集中在最大值）；temperature 越大，权重越接近均匀分布。
- `w`：对 `norm_stat / temperature` 做 softmax 后得到的权重向量，形状 `(L,)`，所有元素之和为 1。例：norm_stat=[2.4, 3.0, 2.8]，temperature=1，则 `w ≈ [0.25, 0.42, 0.33]`。
- `id_emb_i`：第 `i` 个实体 ID 在 Embedding 表里查到的向量，形状 `(D,)`。
- `repr`：最终的字段表示，是对 L 个 `id_emb` 按权重加权求和的结果，形状 `(D,)`。

### 为什么更好

想象有三个实体：A（stat=1）、B（stat=50）、C（stat=2）。均值池化下，A、B、C 贡献相等——B 明明是用户最偏好的实体，它的信号在表示向量里只占 1/3。加权池化后，B 的权重远大于 A 和 C，最终向量更接近 B 的 embedding，准确反映了用户的真实偏好重心。

相比 V6：V6 在**编码每个 embedding** 时用了 stat（让 pair_emb 更好），但**聚合阶段**仍然均值。本方案在聚合阶段引入 stat，两者作用在不同环节，可以叠加使用。

### 成本

几乎无额外参数（`temperature` 是一个标量）。实现只需在池化前加一行 softmax。是所有方案里成本最低的改进。

---

## 方案二：元素级 FiLM 调制（Pair-level FiLM）

### 原理

V6 用的是"加法"：`id_emb + MLP(...)` ，stat 对 id_emb 的影响是加一个常量修正。FiLM 换成"乘法 + 加法"：stat 先经过 MLP 生成两个向量 γ 和 β，然后用 γ 对 id_emb 的每个维度**独立缩放**，用 β 做**偏移**。

```python
norm_stat_i = sign(x_i) * log1p(|x_i|)         # (标量) 对第 i 个 stat 归一化
gamma_i, beta_i = MLP(norm_stat_i)              # 各 (D,)，MLP 权重零初始化
pair_emb_i = id_emb_i * (1 + gamma_i) + beta_i # (D,)  FiLM 调制
repr = mean_pool(pair_emb_i)                    # (D,)  均值池化（或结合方案一做加权池化）
```

**各量说明：**

- `norm_stat_i`：第 `i` 个实体对应的 stat 经过 log 归一化后的标量。
- `MLP(norm_stat_i)`：一个小型全连接网络，输入是 1 维标量，输出是 `2D` 维向量，然后拆成两半。
- `gamma_i`：MLP 输出的前 D 个数，形状 `(D,)`，叫做**缩放向量**（scale）。它决定对 `id_emb_i` 的每一个维度乘以多少倍。
- `beta_i`：MLP 输出的后 D 个数，形状 `(D,)`，叫做**偏移向量**（shift）。它决定对 `id_emb_i` 的每一个维度加上多少偏移。
- `(1 + gamma_i)`：这里写成 `1+γ` 而不是直接用 `γ`，是因为 MLP **零初始化**（所有权重初始为 0），训练刚开始时 `gamma=0`，`(1+0)=1`，乘以 1 等于没有调制——模型从"完全无操作"出发，训练稳定，不会一开始就乱掉。随着训练进行，γ 和 β 逐渐学到有意义的值。
- `pair_emb_i`：经过 FiLM 调制后的第 i 个 pair embedding，形状 `(D,)`。

**具体例子（D=4）：**

```
id_emb_i = [0.5,  1.2, -0.3,  0.8]
gamma_i  = [0.2, -0.1,  0.5,  0.0]   # stat 高 → 第3维放大 50%，第2维缩小 10%
beta_i   = [0.0,  0.1,  0.0, -0.1]

pair_emb_i = [0.5*(1+0.2)+0.0,  1.2*(1-0.1)+0.1,  -0.3*(1+0.5)+0.0,  0.8*(1+0.0)-0.1]
           = [0.60,              1.18,              -0.45,              0.70]
```

### 为什么更好

V6 加性 MLP 的问题：无论 stat 是多少，`MLP([id_emb, stat])` 输出的修正量对 id_emb 所有维度的影响都"加同一个值"，无法差异化地强调或压制某些维度。

FiLM 的优势：γ 是一个 D 维向量，可以对 id_emb 的每个维度独立控制缩放比例。比如：

- stat 高的实体 → γ 让"兴趣强度"相关的维度放大，让"不确定性"相关的维度缩小
- stat 低的实体 → γ 接近全 0，等价于"不调制"（保留原始 embedding）

这比"加一个常量修正"表达力强得多。类比：V6 是"给每个科目统一加 5 分"，FiLM 是"根据每科情况分别乘以不同系数再加分"。

### 成本

每个 fid 一个小 MLP（输入 1，输出 2D）。参数量约 `2D` 个，极小。

---

## 方案三：排名位置编码（Rank-based Positional Encoding）

### 原理

不直接用 stat 的绝对值，而是把实体按 stat 从大到小排序，然后给"排名第几"附加一个**可学习的位置 embedding**，让模型知道"排名第 1 的实体"和"排名第 3 的实体"是不同语义的位置。

```python
order = argsort(stat, descending=True)      # (L,)   按 stat 降序排列的下标
id_embs_sorted = id_embs[order]             # (L, D) 把 id_emb 按 stat 从大到小重排
pos_embs = position_embedding(arange(L))    # (L, D) 取出排名 0,1,...,L-1 的位置向量
pair_emb_i = id_embs_sorted_i + pos_embs_i # (L, D) 每个实体 emb 加上其排名对应的位置 emb
repr = mean_pool(pair_emb_i)               # (D,)
```

**各量说明：**

- `stat`：原始统计值列表，如 `[10.5, 20.0, 15.5]`。
- `order = argsort(stat, descending=True)`：返回排序后的下标。例：stat=[10.5, 20.0, 15.5]，则 order=[1, 2, 0]（20.0 最大排第一，下标是 1；15.5 其次，下标是 2；10.5 最小，下标是 0）。
- `id_embs_sorted`：把原始 `id_embs` 按 order 重新排列，排出来的结果就是"stat 第一名的实体 embedding 在第 0 行，第二名在第 1 行..."。
- `position_embedding`：一个可学习的 Embedding 表，有 `L_max`（最大列表长度）行，每行是 D 维向量。第 0 行表示"排名第 1 的语义位置"，第 1 行表示"排名第 2"，依此类推。这张表的参数量是 `L_max × D`，由训练数据学出。
- `pos_embs = position_embedding(arange(L))`：取出下标 `[0, 1, ..., L-1]` 对应的位置向量，形状 `(L, D)`。
- `pair_emb_i = id_embs_sorted_i + pos_embs_i`：第 i 名的实体 embedding 加上"排名第 i"的位置向量。于是模型在看到这个向量时，不仅知道"这是哪个实体"，还知道"这个实体在该用户中排名第 i"。

### 为什么更好

绝对 stat 值跨用户不可比：用户 A 最大 stat 是 500，用户 B 最大 stat 是 5。归一化后两者的值差别很大，但"排名第一"的含义相同：都是该用户最关注的实体。

排名是**序数信号**，跨用户一致：rank 1 永远意味着"最重要"，和 stat 的绝对大小无关。模型只需要学会"rank 1 的实体权重更高"这一规律，就能泛化到所有用户。

与方案一/二的互补性：方案一/二捕捉"绝对强度"（stat 数值越大越重要），方案三捕捉"相对排序"（谁比谁更重要）。两类信号互补，叠加使用效果更好。

### 成本

排序操作计算量极小（L 通常 < 20）。位置 embedding 表参数量 `L_max × D`，很小。

---

## 方案四：相对统计值编码（Within-List Relative Normalization）

### 原理

不用绝对归一化的 stat，而是计算每个实体的 stat 在**列表内部**的占比（相对权重）或相对排名，把这个相对信号作为额外特征输入 MLP。

```python
# 方式 A：相对权重（每个实体的 stat 占列表总 stat 的比例）
rel_w_i = stat_i / (sum(stat) + eps)             # (L,)  ∈ [0,1]，全部加起来等于 1

# 方式 B：归一化排名（排名越靠前，值越小）
rel_rank_i = rank_i / len(stat)                  # (L,)  ∈ [0,1]

# 把相对信号和绝对信号拼在一起，输入 MLP
pair_input_i = [id_emb_i, norm_abs_stat_i, rel_w_i]  # (D + 1 + 1,) 三类信息拼接
pair_emb_i = MLP(pair_input_i)                   # (D,)
repr = mean_pool(pair_emb_i)                     # (D,)
```

**各量说明：**

- `stat_i`：第 `i` 个实体的原始统计值，如 20.0。
- `sum(stat)`：列表里所有实体 stat 的总和，如 10.5+20.0+15.5=46.0。
- `eps`：极小常数（如 1e-9），防止 stat 全为 0 时分母为零。
- `rel_w_i = stat_i / (sum(stat) + eps)`：第 i 个实体贡献的"占比"。例：stat=[10.5, 20.0, 15.5]，sum=46，rel_w=[0.228, 0.435, 0.337]。rel_w 的所有元素加起来等于 1，相当于把 stat 转成了一个概率分布。
- `rank_i`：第 i 个实体按 stat 降序后的排名（从 0 开始）。stat 最大的排名 0，最小的排名 L-1。
- `rel_rank_i = rank_i / len(stat)`：归一化排名，值域 [0, 1/(L-1)/L ... (L-1)/L]。
- `norm_abs_stat_i`：用 log1p 归一化后的绝对 stat，是第 i 个实体 stat 的"全局尺度"信息。
- `pair_input_i`：把三类信息拼在一起——**"是什么实体"**（id_emb）、**"绝对有多重要"**（norm_abs_stat）、**"相对有多重要"**（rel_w）——作为 MLP 的完整输入。

### 为什么更好

假设有两个用户：

- 用户 A：stat=[50, 49, 51]（三个实体几乎均等，rel_w ≈ [0.33, 0.33, 0.33]）
- 用户 B：stat=[100, 1, 1]（第一个实体绝对主导，rel_w ≈ [0.98, 0.01, 0.01]）

`rel_w` 直接揭示了**用户兴趣是分散的还是集中的**这一关键信息。如果只看绝对 log1p 值，模型很难发现这种结构差异；而 rel_w 让这种差异一目了然，模型可以学到"兴趣集中的用户和兴趣分散的用户有不同的行为规律"。

### 成本

纯数学计算，无额外参数。把 `rel_w` 拼进 MLP 输入只增加一列，几乎没有成本。

---

## 方案五：字段内自注意力（Intra-field Self-Attention）

### 原理

前四个方案都是对每个 (id, stat) 对**独立编码**，然后聚合。本方案把同一个 fid 内的所有 pair 当作一个**微型 sequence**，先用 Self-Attention 让它们**相互感知**，再聚合。

```python
# Step 1：把每个 (id_emb, norm_stat) 对融合成一个 token
tokens = [MLP([id_emb_i, norm_stat_i]) for i in range(L)]  # L 个 (D,) 向量，构成 (L, D)

# Step 2：在 L 个 token 之间做 Self-Attention，让每个实体"看到"同字段的其他实体
attended = self_attention(tokens)   # (L, D)，Q=K=V=tokens

# Step 3：池化得到字段表示
field_repr = mean_pool(attended)    # (D,)
```

**可选扩展（DIN 风格）**：用目标 item 做 Query，实现"针对这次曝光，哪些历史实体最相关"的个性化聚合：

```python
query = item_emb(target_item_id).unsqueeze(0)        # (1, D)  目标广告 item 的 embedding
field_repr = cross_attention(Q=query, K=tokens, V=tokens)  # (1, D)
```

**各量说明：**

- `MLP([id_emb_i, norm_stat_i])`：把第 i 个实体的 embedding（D 维）和其归一化 stat（1 维）拼接后过 MLP，输出 D 维的 token。这一步先做基础的 pair 融合。
- `tokens`：L 个 token 组成的矩阵，形状 `(L, D)`，每行代表一个实体的初始表示。
- `self_attention(tokens)`：对 `tokens` 做多头 Self-Attention。具体地，Q=K=V 都是 `tokens`，attention 的计算方式是每个 token 去和所有其他 token 计算相关性得分，然后按得分加权求和。输出形状仍然是 `(L, D)`，但每个位置的向量已经融入了其他实体的信息。
- `attended`：Self-Attention 后的结果，形状 `(L, D)`。每个位置的向量现在"知道"同字段中有哪些其他实体，以及它们的 stat 大小。
- `field_repr = mean_pool(attended)`：对 L 个 attended 向量取均值，得到字段级表示，形状 `(D,)`。

**DIN 扩展中：**

- `target_item_id`：当前这条样本里，用户正在看的目标广告的 ID（标量）。
- `item_emb(target_item_id)`：目标广告的 embedding，形状 `(D,)`。
- `query = ....unsqueeze(0)`：在第 0 维增加一个维度，变成 `(1, D)`，满足 cross_attention 接口的 Query 输入格式。
- `cross_attention(Q=query, K=tokens, V=tokens)`：用目标 item 作为 Query，tokens 作为 Key 和 Value，计算目标 item 和每个历史实体的相关性，然后加权聚合 tokens。结果是"这个广告最关心用户哪些历史实体"的综合表示。

### 为什么更好

前四个方案把每个实体独立对待，无法捕捉**实体之间的关系**。但用户行为本身是有结构的：

> "用户对实体 A 停留时长高，对相关实体 B 也高" → 这两个实体可能属于同一品类，形成了一个强兴趣簇

Self-Attention 让实体 A 的编码受到实体 B 的影响，反之亦然。模型可以学到"当 A 和 B 同时高 stat 时，说明用户有某种集中偏好"这样的组合规律，而独立编码永远无法发现这一点。

DIN 风格的 Cross-Attention 更进一步解决了另一个问题：对于不同的目标广告，用户历史里不同的实体重要程度不同（看运动广告时，运动品类的历史比食品品类更相关）。Cross-Attention 让聚合结果和目标广告挂钩，实现**对当前曝光上下文敏感的用户表示**。

### 成本

Self-Attention 复杂度 O(L²D)，但 L 通常小于 20，代价很小。引入参数量约 O(D²)。

---

## 方案六：跨字段交互（Cross-field Interaction）

### 原理

fid 62-66 是五个字段，89-91 是三个字段。目前每个字段各自独立编码，互不相知。本方案把同组的字段编码后的向量**再做一次交互**，让模型能发现跨字段的组合规律。

```python
# Step 1：先用前面某个方案，把每个字段编码成一个向量
field_reprs = [encode(fid_62), encode(fid_63), encode(fid_64), encode(fid_65), encode(fid_66)]
# 5 个 (D,) 向量

field_tokens = stack(field_reprs)               # (5, D)  把 5 个向量叠成矩阵

# 方式 A：简单拼接 + MLP（最快）
cross_repr = MLP(field_tokens.flatten())        # (5D,) → MLP → (D,)

# 方式 B：字段间 Self-Attention（更强）
cross_repr = mean_pool(self_attention(field_tokens))  # (5, D) → Attn → (5, D) → pool → (D,)
```

同理，对 89-91 的三个字段做 `(3, D)` 的跨字段交互。

**各量说明：**

- `encode(fid_62)`：用前面某个方案（比如方案一或方案二）对 fid 62 内部的 (id, stat) 对做聚合，得到 `(D,)` 的字段表示向量。
- `field_tokens = stack([...])`：把 5 个 `(D,)` 向量堆叠成 `(5, D)` 的矩阵。相当于把"5 个字段的表示"视为"5 个 token"，方便后续做 Attention。
- `field_tokens.flatten()`：把 `(5, D)` 展平成 `(5D,)` 的一维向量，作为 MLP 的输入。MLP 会学习 5 个字段之间的任意组合关系。
- `self_attention(field_tokens)`：让 5 个字段 token 互相做 Attention。字段 62 可以"看到"字段 65 的内容，63 可以"看到"64，等等。每个字段的最终表示都融入了对其他字段的感知。
- `mean_pool(...)`：对 5 个 attended 字段 token 取均值，得到整个组的综合表示，形状 `(D,)`。

### 为什么更好

fid 62-66 很可能描述的是同一类用户行为的不同侧面（不同品类层级、不同时间窗口的统计等）。用户在不同字段上的偏好**联合分布**往往比任意单个字段更有预测力：

> "fid_62 里偏好某实体 + fid_63 里也偏好相关实体" → 这种跨字段的共现模式比单字段更能定位用户的深层兴趣

这和 DCN、DeepFM 的 feature crossing 思路相同，只不过这里的 feature 是"用户在某字段内的聚合偏好向量"，而不是原始的 one-hot。

目前所有方案都把 8 个 fid 完全独立处理，跨字段的组合信号被丢弃。

### 成本

字段数固定（5 或 3），Self-Attention 代价 O(25D) 或 O(9D)，极小。

---

## 方案比较总结

| 方案 | 主要解决的问题 | 参数量 | 实现复杂度 | 与现有 V6 的关系 |
|------|--------------|--------|----------|----------------|
| 一：stat 加权池化 | 聚合时忽略重要性差异 | 极少（1 个标量） | 低 | 可叠加在 V6 之上 |
| 二：Pair-level FiLM | 加性交互表达力不足 | 小（每 fid 一个小 MLP） | 低 | 替代 V6 的 residual MLP |
| 三：排名位置编码 | 绝对值跨用户不可比 | 小（L_max × D 的 emb 表） | 低 | 与方案一/二互补 |
| 四：相对归一化 | 列表形态（集中/分散）未建模 | 无 | 极低 | 作为 MLP 额外输入，无缝叠加 |
| 五：字段内 Self-Attn | 实体间关系、target-aware 聚合 | 中（O(D²)） | 中 | 替代均匀 pool |
| 六：跨字段交互 | 62-66 / 89-91 组内字段无交互 | 小 | 中 | 加在各字段 field repr 上 |

---

## 推荐实验路径

优先级从高到低：

1. **方案一 + 方案四**（最低成本，无新参数）：用相对 stat（rel_w）拼接绝对 stat 后做加权池化，快速验证 stat 信号是否被充分利用。

2. **方案二（Pair-level FiLM）**：替换 V6 的 residual MLP，增强 stat 对 embedding 的调制能力。零初始化保证训练稳定。

3. **方案三（排名位置编码）** + **方案二叠加**：把 FiLM（绝对强度）+ 排名（相对顺序）两种信号同时注入，互补。

4. **方案五（字段内 Self-Attention）**：需要更多调试，但上限可能更高。建议先跑通 1-3 之后再试。

5. **方案六（跨字段交互）**：配合方案五一起实验，看 62-66 / 89-91 的字段间联合建模是否有额外增益。