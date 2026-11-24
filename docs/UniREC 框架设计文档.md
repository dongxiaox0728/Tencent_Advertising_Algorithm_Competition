# UniREC 框架设计文档

**KDD Cup 2026 · TAAC · Round 1 starts Apr 24**

------

## 1. 任务定义

给定用户历史行为序列 S 和静态属性 NS，预测对候选广告的转化概率（二分类，BCE loss，AUC 评估）。

**输入：**

- S（Sequential）：`action_seq` / `content_seq` / `item_seq`，变长序列，最长 4000
- NS（Non-Sequential）：`user_feature`（用户静态属性）+ `item_feature`（候选广告特征，含 target item）

**输出：** 转化概率 ∈ [0, 1]

**关键约束：**

- target item 从不出现在历史序列中（纯泛化任务，不能依赖"用户看过这个广告"的信号）
- 数据全脱敏（整数 ID，无原始语义）
- 禁止 ensemble
- 推理延迟硬限制（Industrial Track 更严格）

------

## 2. UniRECBlock 结构

```
输入: s [B, L_s, D]  ·  ns [B, L_ns, D]  ·  target [B, 1, D]
        ↓
  Step 1: S internal — Masked Self-Attention + FFN
        ↓
  Step 2: NS internal — Per-Token FFN
        ↓
  Step 3: S × NS — Adaptive Bidirectional Cross-Attention（Vector Gate）
        ↓
输出: s' [B, L_s, D]  ·  ns' [B, L_ns, D]
```

**设计理由：** 先做 Step 1 和 Step 2，让 S 和 NS 各自形成稳定表示，再用这些"成熟"的表示做跨类型交互。Gate 信号质量依赖于输入表示的质量，提前交互会降低 gate 的判断准确性。

### Step 1：S internal

Masked Self-Attention + FFN，捕捉序列内时序依赖。`causal_mask` 可配置（autoregressive 或 bidirectional）。

### Step 2：NS internal

Per-Token FFN，每个字段独立变换，不做跨字段 attention。

理由：NS 各字段语义完全异构（地区、年龄、兴趣标签等），跨字段 self-attention 中的 inner-product similarity 没有物理意义。这和 OneTrans 中 NS-token 需要独立参数的理论依据一致——异构字段强行共享参数会导致 capacity collapse。

### Step 3：Adaptive Bidirectional Cross-Attention

见下节详细说明。

------

## 3. 自适应 Gate 设计（核心创新）

### Gate 计算（target-aware）

```python
# 用 target item 对 S 和 NS 分别做轻量 cross-attention，得到 target-aware 摘要
h_s  = CrossAttn(q=target, k=s,  v=s ).squeeze(1)   # [B, D]
h_ns = CrossAttn(q=target, k=ns, v=ns).squeeze(1)   # [B, D]

# element-wise product 捕捉每个语义维度上两侧的一致性或分歧
gate_input = cat([h_s, h_ns, h_s ⊙ h_ns])           # [B, 3D]
g = sigmoid(Linear(gate_input)).unsqueeze(1)          # [B, 1, D]
```

### 更新公式

```python
s_cross  = MultiHeadAttn(q=s,  k=ns, v=ns)   # S 从 NS 提取信息: [B, L_s, D]
ns_cross = MultiHeadAttn(q=ns, k=s,  v=s)    # NS 从 S 提取信息: [B, L_ns, D]

s'  = LayerNorm(s  + dropout(g       * s_cross))
ns' = LayerNorm(ns + dropout((1 - g) * ns_cross))
```

`g ≈ 1`：S 大量吸收 NS 的静态先验，NS 侧更新被压制。
 `g ≈ 0`：NS 大量吸收 S 的近期行为信号，S 侧更新被压制。
 方向由当前样本内容动态决定，不是固定参数。

### 为什么不用 mean pool 驱动 gate（与初版的区别）

初版用 `s.mean(dim=1)` 驱动 gate，问题是 mean pool 对所有候选广告返回同一个值——它不知道当前候选广告是什么，无法做 query-aware 的自适应判断。

改进后，gate 同时用 target 对 S 和 NS 做 cross-attention，得到「对于这个广告，S 里最相关的信息」和「对于这个广告，NS 里最相关的信息」，再做对比。Gate 真正在回答「这个广告更需要从 S 还是 NS 吸收信息」。

------

## 4. Prediction Head

```python
# S 侧：target 作 query，提取 target-aware 的最终 S 表示
s_pool  = CrossAttn(q=target, k=s, v=s).squeeze(1)  # [B, D]
# NS 侧：mean pool（NS 字段数量少，信息密度均匀，mean pool 损失有限）
ns_pool = ns.mean(dim=1)                             # [B, D]

logit = MLP(cat([s_pool, ns_pool]))                  # [B, 1]
```

和 HyFormer Query Decoding 的思想一致：用带着全局信息的 query 去读序列，而不是盲目平均。S 最终表示由候选广告驱动，提取的是"和这个广告最相关的历史行为信息"。

------

## 5. 三条序列处理方案

### 推荐方案：concat + segment embedding

```python
# 三条序列直接 concat，加 learnable segment embedding 区分来源
s = cat([action_seq, content_seq, item_seq], dim=1)  # [B, L_total, D]
s = s + segment_emb[seq_type]                         # seq_type ∈ {0, 1, 2}，per-token
```

**理由：** 三条序列本身有关联（同一用户不同维度的行为），隔离太彻底反而损失跨序列信号。不单独做三个独立 encoder，原因是参数量和复杂度会成倍增加，收益不明确。Segment embedding 让模型自己学如何利用不同类型序列的信息。

**待消融验证：** concat + segment_emb vs 三路独立 encoder vs 直接 concat 不加区分。

------

## 6. 整体 Backbone

```
Embedding 层（待对接 baseline 数据 pipeline）
        ↓
UniRECBlock × N 层（可堆叠）
        ↓
target-aware S pool + NS mean pool → concat
        ↓
MLP → logit（BCE loss 训练）
```

N 层数量、`d_model`、`d_ff`、`causal_mask` 均通过 config 控制，方便做 scaling 实验。

------

## 7. 与已读文章的横向对比

### 各方案定位一览

| 文章        | 机构             | 核心问题                                       | 范式                                    |
| ----------- | ---------------- | ---------------------------------------------- | --------------------------------------- |
| RankMixer   | ByteDance/Douyin | 序列 + NS 特征交互的 GPU 效率与 scaling        | encode-then-interact（单向，两阶段）    |
| HyFormer    | ByteDance/Douyin | 序列建模与特征交互的双向耦合                   | 统一骨干，每层双向信息流                |
| OneTrans    | ByteDance/Douyin | 统一 token 化，S 和 NS 放入同一 self-attention | 统一 self-attention，参数异构           |
| InterFormer | —                | 多场景推荐，summarization vector 做场景间迁移  | 两阶段，summarization vector 作信息瓶颈 |
| LONGER      | ByteDance/Douyin | 端到端超长序列建模（序列长度 → 10000）         | 序列压缩 + hybrid attention，工程优先   |
| Wukong      | Meta AI          | 纯 NS 特征交互的 scaling law                   | 堆叠 FM + 线性压缩，无序列建模          |
| **本方案**  | —                | S 与 NS 的类型感知双向交互，target-aware gate  | 双向 cross-attention + vector gate      |

### 关键维度对比

| 维度              | RankMixer                                   | HyFormer                                           | OneTrans                                         | InterFormer                           | LONGER                                     | Wukong                       | **本方案**                                       |
| ----------------- | ------------------------------------------- | -------------------------------------------------- | ------------------------------------------------ | ------------------------------------- | ------------------------------------------ | ---------------------------- | ------------------------------------------------ |
| S/NS 交互方式     | Token Mixing（参数无关，reshape-transpose） | Global Token 枢纽，Query Decoding + Query Boosting | 统一 self-attention，所有 token 平等参与         | Summarization vector 作瓶颈，单向传递 | Concat + MLP，序列压缩后拼接 NS            | 无序列，仅 NS 特征的 FM 堆叠 | 双向 cross-attention，类型感知                   |
| 信息流方向        | 单向（序列 → 交互）                         | 每层双向交替迭代，并非同时交互                     | 对称，所有 token 互相 attend                     | 单向（序列 → summarization → NS）     | 单向（序列压缩 → 拼接）                    | 无序列                       | 双向，gate 自适应控制权重                        |
| Gate / 自适应机制 | 无                                          | 无（权重由 attention 学习）                        | 无                                               | 无                                    | 无                                         | 无                           | target-aware vector gate（双侧 cross-attn 驱动） |
| NS 内部处理       | per-token FFN                               | MLP-Mixer（跨 token 子空间混合）                   | self-attention（独立参数，防 capacity collapse） | self-attention                        | Concat + MLP                               | FM 显式二阶交互              | per-token FFN                                    |
| S 内部处理        | LONGER 独立外部模块                         | 多条序列独立 cross-attn，Global Token 作 Q         | self-attention（causal mask）                    | self-attention                        | self-attention + Token Merge（InnerTrans） | 无                           | masked self-attention（可配置 causal）           |
| 长序列处理        | 依赖外部压缩（LONGER）                      | 支持长序列（MeanPool 冷启动 query）                | KV Cache + Pyramid Stack 降维                    | 不处理                                | 核心贡献（Token Merge，~50% FLOPs 节省）   | 无                           | 未专项优化（待对接截断策略）                     |
| Prediction head   | 未明确（外部 MLP）                          | 顶层 Global Token → MLP                            | 顶层 token → MLP                                 | 顶层输出 → MLP                        | Concat → MLP                               | 顶层输出 → MLP               | target-query cross-attn → MLP                    |
| Scaling 验证      | 有（FLOPs/参数/层数四轴）                   | 有（曲线斜率优于 BaseArch）                        | 未明确                                           | 无                                    | 有（序列长度 scaling）                     | 有（核心贡献，ICML 2024）    | 无（待实验）                                     |
| 工业部署          | 是（Douyin，1B 参数）                       | 是（Douyin 搜索，数十亿用户）                      | 是（字节系）                                     | 未明确                                | 是（Douyin，dozens of 场景）               | Meta 内部验证                | 目标：KDD Cup Round 1                            |

### 本方案的设计谱系

```
RankMixer   ──→  per-token FFN 用于 NS 内部（借鉴）
                  Token Mixing 不采用（异构 token inner-product 无意义）

HyFormer    ──→  Query Decoding 思想（用 target 作 query 读序列）
                  每层双向信息流（借鉴）
                  MLP-Mixer 不采用（换成 vector gate 更直接）

OneTrans    ──→  NS token 独立参数思路（异构字段不共享参数）
                  统一 self-attention 不采用（S/NS 语义差异大，对等交互引入噪声）

InterFormer ──→  反面教材：summarization vector 信息瓶颈 + 架构碎片化，不采用

LONGER      ──→  Token Merge 思想可后续引入处理超长序列（当前待对接）
                  Hybrid attention 的工程优化值得参考

Wukong      ──→  NS 字段间 FM 二阶交互（低优先级消融项）
                  Scaling law 实验设计思路（借鉴）
```

------

## 8. 消融实验计划

### 🔴 高优先级（baseline 一出来立刻跑）

**A. Gate 驱动方式**

| 条件              | 描述                                              |
| ----------------- | ------------------------------------------------- |
| baseline          | S 和 NS 都用 mean pool 驱动 gate（初版）          |
| S-aware only      | 仅 S 侧用 target cross-attention，NS 侧 mean pool |
| NS-aware only     | 仅 NS 侧用 target cross-attention，S 侧 mean pool |
| 双侧 target-aware | S 和 NS 都用 target cross-attention（完整版）     |

验证：NS 侧 target-aware 是否有额外贡献；mean pool 的信息损失有多严重。

**B. 三条序列合并方式**

| 条件                 | 描述                         |
| -------------------- | ---------------------------- |
| concat + segment_emb | 推荐方案                     |
| 三路独立 encoder     | 各自过 self-attention 后合并 |
| 直接 concat          | 不加 segment embedding       |

验证：segment embedding 是否足够；独立 encoder 的参数开销是否值得。

**C. Prediction head**

| 条件                    | 描述   |
| ----------------------- | ------ |
| mean pool               | 基线   |
| target-query cross-attn | 本方案 |

验证：最终表示提取方式的影响；与 gate 改进是否有叠加效果。

### 🟡 中优先级

**D. Gate 对称性**

| 条件        | 描述                                  |
| ----------- | ------------------------------------- |
| 单 gate     | g 控制 S 侧，(1-g) 控制 NS 侧（当前） |
| 独立双 gate | g_s 和 g_ns 各自独立学习              |

验证：两方向是否需要独立控制；参数增加是否带来收益。

**E. Causal mask**

| 条件          | 描述                     |
| ------------- | ------------------------ |
| causal        | autoregressive，只看历史 |
| bidirectional | 全序列双向 attention     |

验证：序列 attention 方向对广告转化预测任务的影响（非生成任务，bidirectional 可能更合适）。

### 🟢 低优先级（前面结论稳定后再动）

**F. NS 内部交互**

| 条件              | 描述                                        |
| ----------------- | ------------------------------------------- |
| per-token FFN     | 当前方案，无跨字段交互                      |
| FFN + FM 二阶交互 | 显式建模字段间 pairwise 交互（参考 Wukong） |
| self-attention    | 允许跨字段动态交互                          |

验证：字段间二阶交互是否有额外收益；FM 的归纳偏置 vs FFN 的通用性。

**⚠️ 消融注意事项：** 控制总参数量，避免 AUC 提升源于参数增多而非设计改进。对于引入额外 attention 模块的条件，适当缩减 `d_model` 或 `n_heads` 进行参数补偿。

------

## 9. 待完成（baseline 释放后）

- [ ] **Embedding 层**：对接 baseline 数据 pipeline；构造 target item 独立 token；处理 `int_array` 稀疏特征的 embedding lookup
- [ ] **Padding mask**：变长序列的 attention mask，防止 pad token 参与 softmax 计算
- [ ] **序列截断策略**：三条序列最长 4000，确定截断长度（保留最新 N 条）及 concat 后的总长度上限
- [ ] **None / 空字段处理**：`user_feature` 存在少量缺失字段，需要 mask 或补零策略
- [ ] **推理延迟测量**：在 baseline 环境下测量 block 数量 / `d_model` 对延迟的影响，确保满足 Industrial Track 硬限制