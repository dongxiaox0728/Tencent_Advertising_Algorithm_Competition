# PCVRHyFormer 迭代记录

记录在 `baseline_amp_with_sincos_time_feature` 上从 `0.8146` 一路打到 `0.8212+` 的所有**已验证有效**改动、对应原理、失败实验（避坑），以及当前完整的训练 / 推理流程。

## 1. 总览

### 1.1 测试集 AUC 演进

| # | 改动 | Test AUC | Δ | 类型 |
|---|---|---|---|---|
| 0 | 基线（standalone time NS token + per-channel gate） | 0.8146 | — | — |
| 1 | + **FiLM γ/β 调制** | 0.81485 | +0.00025 | 架构 |
| 2 | + **时间衰减加权 pooling** | 0.8149 | +0.00005 | 架构 |
| 3 | + **用户级 staleness** | **0.8194** | **+0.0045** | 信号 |
| 4 | + **Cosine LR + warmup** | **0.8212** | **+0.0018** | 训练侧 |
| 5 | + **EMA (decay=0.999)** + Cosine 全周期 | 跑中 | — | 训练侧 |

### 1.2 失败的实验

| 改动 | 性质 | 损失 |
|---|---|---|
| K=4 多阶傅立叶谐波 | 增加时间投影器输入维度 | -0.016 |
| 序列事件随机 mask (p=0.15) | 训练时丢弃信号 | -0.001 |
| Per-domain staleness | 1 张表拆成 S 张（同源信号） | -0.010 |
| Per-domain activity count | 加 S 张 count embedding（与 seq_lens 强相关）| -0.021 |

### 1.3 核心观察

- **加信号 vs 加容量**：模型一旦出现"加容量就跌"的现象，说明这条信号源已经饱和；只有**真正新维度**的信号会涨（staleness 是 per-sample 全局摘要，之前完全不存在）。
- **训练侧优化 vs 特征侧**：当特征侧饱和之后，cosine LR schedule 立刻又给了 +0.0018 ——证明模型现在已经从"加信号"阶段进入"训训练好"阶段。
- **单变量纪律**：每次只动一个东西。第一次叠了 K=4 + decay_pool 同时改动，掉点之后无法归因，浪费了一次实验。

---

## 2. 已验证有效的改动（原理 + 为什么有用）

### 2.1 FiLM 调制（Feature-wise Linear Modulation）

**位置**：[`TimeFeatureEmbedder`](model.py) 内部新增 `film_proj: Linear(d_model → 2*d_model)`，在 [`_build_representations`](model.py) 里应用到所有 NS tokens。

**机制**：

```
time_emb → film_proj → (γ, β)     # 每个 d_model 维度都有自己的 (γ, β)
ns_token' = ns_token * (1 + γ) + β  # 逐通道 affine
```

**原理**：经典的条件调制。把时间从一个**旁观的 NS token** 升级成**主动 reshape 其它 NS tokens** 的全局上下文。同一个 user_ns 在凌晨 2 点和晚上 8 点会被解读成完全不同的表示。

**为什么有用**：
- 旧方案是把时间 token concat 到 NS pool 里，模型要从零学"时间-用户特征"的交叉，而 attention/RankMixer 这一层学这种乘法交叉本来就不擅长
- FiLM **把乘法交叉写进架构**，让时间天生有权力 reshape 任意被调制的特征
- **零初始化** `film_proj` 保证起点等价于无 FiLM 版本，模型只能向上不向下

### 2.2 时间衰减加权 pooling

**位置**：[`MultiSeqQueryGenerator._pool_seq`](model.py)。

**机制**：原本对序列 token 做简单 mean pool，现在改成：

```
τ_i = exp(log_tau[seq_idx])                   # per-sequence 可学习
weight_j = exp(-bucket_id_j / τ_i)             # 越近的事件权重越高
pooled = Σ_j (token_j * weight_j) / Σ_j weight_j
```

每条序列 domain 有自己的 `log_tau`，初始化 `τ = num_buckets / 2 ≈ 32`（中性起点）。

**为什么用 bucket_id 而不是真实秒数**：
- bucket_id 已经是 log-spaced 的（5s, 10s, ..., 1 年覆盖 64 桶）
- `exp(-bucket_id/τ)` 在真实时间轴上其实是**幂律衰减**的近似，更符合用户行为遗忘规律
- 数值稳定（范围 0–64 而不是 0–31536000）

**为什么有用**：
- mean pool 把"5 秒前的点击"和"1 年前的点击"当成同等重要，信号被稀释
- 让模型自己学每条序列的"半衰期"（点击序列可能 τ 小，转化序列 τ 大）

### 2.3 用户级 staleness（最大单笔涨幅 +0.0045）

**位置**：[`TimeFeatureEmbedder`](model.py) 内部加 `staleness_emb: nn.Embedding(num_time_buckets, d_model)`，在 [`_build_representations`](model.py) 算 per-sample 全局最近活动 bucket id。

**机制**：

```
# 跨所有序列 domain 取每个用户最近一次活动的 bucket id
user_staleness_bucket = min over (所有 domain, 所有 valid 事件) of bucket_id_j

# 嵌入并加到时间 token 上（作为残差，零初始化）
token = self.proj(time_features) + staleness_emb(user_staleness_bucket)
# token 然后进 gate 和 FiLM 两条路径，所以 staleness 同时影响这两条
```

**为什么这是最大的单笔涨幅**：

之前模型有 per-event bucket embedding（每个事件自带 staleness）和 decay pool（事件级权重），但**缺一个 per-sample 的全局摘要**："这个用户作为整体有多新鲜？" bucket=1 意味着用户刚刚还活跃（高 CVR 意图），bucket=64 意味着用户一年没动过（冷启动）。

这是一个**真正缺失的信号轴**，不是已有信号的细化。

**为什么不能用新 NS token 实现**：
- `rank_mixer_mode='full'` 要求 `d_model % T == 0`，当前 T=16，64%16=0，再加 1 个 token 变 17 就破坏约束
- 解法：折进 `TimeFeatureEmbedder` 当残差，**不新增 NS token**

**为什么零初始化**：
- 初始 staleness embedding 全为 0，行为等同于无 staleness 版本
- 模型可以选择把它学成有用的值，也可以一直保持 0；只能向上不向下

### 2.4 Cosine LR Schedule + Warmup（+0.0018）

**位置**：[`PCVRHyFormerRankingTrainer.__init__`](trainer.py) 构建 `LambdaLR`，[`_train_step`](trainer.py) 每步调用。

**机制**：仅对 dense optimizer (AdamW) 生效。

```
step < warmup_steps:
    lr = lr_max * step / warmup_steps          # 线性 warmup

warmup_steps ≤ step < decay_total_steps:
    progress = (step - warmup) / (decay_total - warmup)
    lr = lr_max * (lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (1 + cos(π * progress)))

step ≥ decay_total_steps:
    lr = lr_max * lr_min_ratio                 # plateau on min
```

**当前配置**：`warmup=500, decay_total=22000, lr_min_ratio=0.1`（cosine 在训练自然终止时刚好走完）。

**为什么不调度 sparse Adagrad**：
- Adagrad 自身的 per-parameter accumulator 已经隐含衰减
- trainer 还会在 epoch ≥ `reinit_sparse_after_epoch` 之后 rebuild 整个 Adagrad，会让外部 scheduler 失效

**为什么有用**：
- Warmup 避免训练早期 embedding 还不稳定时被大梯度推坏
- Cosine decay 让训练末期 LR 降到很小，模型能精细微调，落到更尖锐的局部最小值
- 经验上对 transformer / recsys 类模型几乎是免费 +0.001~0.003

### 2.5 EMA 权重平均（验证中）

**位置**：[`ModelEMA`](trainer.py) 类 + trainer 集成。

**机制**：

```python
# 每个 optimizer step 之后
shadow = decay * shadow + (1 - decay) * param  # in-place

# Eval 和保存时
swap_in(model)        # 备份 raw 权重，加载 shadow
val_auc = evaluate()
save_checkpoint()      # 保存的 model.pt 就是 EMA 模型
swap_out(model)       # 恢复 raw 权重继续训练
```

**关键设计**：
- **Eval 和 save 都用 EMA 权重**：所以 `model.pt` 直接就是 EMA 模型，**infer.py 零改动**
- **高基数 sparse reinit 后同步 shadow**：否则 EMA 会持有 stale 老值，反而抵消 reinit 的收益
- Default `--ema_decay 0.0`（关闭），需要在 run.sh 显式开启，避免静默影响其它 job

**为什么有用**（理论 + 经验）：
- EMA 等价于在 loss landscape 上做了某种"软平均"，能跳过 SGD 后期的高频抖动
- 等效窗口 `1 / (1 - decay)`：decay=0.999 → ~1000 步窗口
- 竞赛里几乎稳定 +0.001~0.003，特别适合训练曲线有噪声的场景（你这里 test AUC 跨 0.02 量级抖动，正对症）

---

## 3. 失败实验（不要再走的弯路）

### 3.1 K=4 多阶傅立叶谐波（-0.016）

把时间投影器的输入从 6 维（1 阶 sin/cos × 3 周期）扩到 24 维（4 阶谐波 × 3 周期）。**loss 曲线训练侧涨了，test 暴跌**。

**根因**：增加了对训练时间分布的拟合容量，但训练/测试时间分布存在 gap。`train.py` 里 default 已经锁回 K=1。

### 3.2 序列事件随机 mask（-0.001）

训练时对每个 valid 事件按 p=0.15 随机置零 + 加入 padding mask。期望它做正则化。

**根因**：模型不是过拟合，是欠信号。强行丢弃信号反而掉点。这次实验直接帮助诊断出"现在处于训训练好阶段，不是 add signal 阶段"。

### 3.3 Per-domain staleness（-0.010）

把全局 min_bucket 拆成 S 张独立 embedding 表（每个 domain 一张）。

**根因**：per-domain min buckets 大部分时候跟 global min 强相关，4 倍容量没换来新信号。提示：refining 一个 winning 信号的同源信号 ≠ 免费收益。

### 3.4 Per-domain activity count（-0.021，最严重）

按 valid 事件数桶化（log-spaced）再嵌入到时间 token。

**根因**：activity count 跟 `seq_lens` 高度相关，模型已经通过 padding mask 知道这个信息，本质是冗余信号 + 多余容量。

---

## 4. 当前完整训练流程

### 4.1 模型架构（自上而下）

```
Inputs (ModelInput)
├── user_int_feats, item_int_feats      (B, ·)
├── user_dense_feats, item_dense_feats  (B, ·)
├── seq_data: {domain → (B, S, L)}
├── seq_time_buckets: {domain → (B, L)}
└── time_hour / time_day / time_hour_of_week (B,)

═══════════════════════════════════════════════════════════
NS Tokens 构造
═══════════════════════════════════════════════════════════

RankMixerNSTokenizer:
  所有 int features → cat → 平均分成 num_ns_tokens 段 → 每段 Linear → silu → LN
  得到 user_ns: (B, 4, D), item_ns: (B, 2, D)

dense_proj:
  user_dense → Linear+LN → user_dense_token: (B, 1, D)

TimeFeatureEmbedder:
  time_hour/day/week → cyclic sin/cos (K=1 → 6 维) → MLP → token  (B, D)
  token += staleness_emb(user_staleness_bucket)            # ★ 用户级 staleness 残差
  film_proj(token) → (γ, β)                                # ★ FiLM 调制参数
  time_token = sigmoid(gate) * token   (B, 1, D)            # 独立时间 NS token

NS pool (concat 后):
  [user_ns; user_dense_token; item_ns; time_token]
  对前三组应用 FiLM:  ns_part = ns_part * (1 + γ) + β       # ★ FiLM
  得到 ns_tokens: (B, 8, D)

═══════════════════════════════════════════════════════════
Seq Tokens
═══════════════════════════════════════════════════════════

对每个 domain：
  emb(seq) 各 side-info feature → concat → proj → seq_tokens (B, L, D)
  seq_tokens += time_embedding(bucket_id)                  # per-event time bucket

═══════════════════════════════════════════════════════════
Query 生成 + HyFormer Block 堆叠
═══════════════════════════════════════════════════════════

MultiSeqQueryGenerator:
  对每个 domain：
    seq_pooled = Σ(seq_token * exp(-bucket/τ)) / Σ(exp(-bucket/τ))   # ★ decay pool
    global_info = [ns_flat; seq_pooled]
    q_tokens[i] = [FFN_1(global_info), ..., FFN_Nq(global_info)]   (B, Nq, D)

MultiSeqHyFormerBlock × num_hyformer_blocks:
  per domain:
    seq_tokens = SeqEncoder(seq_tokens)               # Transformer / Longer / SwiGLU
    q_tokens = CrossAttention(q_tokens, seq_tokens)
  RankMixerBlock([q_tokens for all domains; ns_tokens]):
    token mixing (full mode, 参数无关 reshape)
    per-token FFN (shared)
    返回更新后的 [q'; ns']

═══════════════════════════════════════════════════════════
输出
═══════════════════════════════════════════════════════════

output_proj(cat all q_tokens) → (B, D)
clsfier: Linear+LN+SiLU+Linear → logit (B, 1)
```

### 4.2 训练循环（trainer.py）

```
for epoch in 1..num_epochs:
    for batch in train_loader:
        loss = forward + bce / focal loss               (with AMP autocast)
        loss.backward + clip_grad_norm(1.0)
        dense_optimizer.step()                          # AdamW (含 LR schedule)
        sparse_optimizer.step()                         # Adagrad (无 schedule)
        lr_scheduler.step()                             # ★ cosine + warmup
        ema.update(model)                               # ★ EMA shadow 更新

        if step % eval_every_n_steps == 0:
            _do_validation():
              ema.swap_in()                             # ★ 切到 EMA 权重
              evaluate() → val_auc
              early_stopping(val_auc, model)           # 保存 EMA 模型
              ema.swap_out()                            # 恢复原权重继续训练

    _do_validation()                                    # epoch 末再评一次

    if epoch >= reinit_sparse_after_epoch:
        reinit_high_cardinality_params()                # 高基数 sparse cold restart
        rebuild Adagrad
        ema.resync_params(reinit_ptrs)                  # ★ 同步 EMA shadow

    if early_stopping.early_stop:
        break
```

### 4.3 优化器

| Optimizer | 参数 | LR | Schedule |
|---|---|---|---|
| AdamW (dense) | model 非 embedding 参数 | 1e-4 | linear warmup 500 → cosine decay → lr_min |
| Adagrad (sparse) | 所有 nn.Embedding | 0.05 | 无（自身自适应，且训练中会 rebuild） |

### 4.4 推理流程（infer.py）

```
1. 读 train_config.json
2. resolve_model_cfg(): 兜底缺失 key，过滤 PCVRHyFormer 不认识的 key  (★ inspect.signature)
3. PCVRHyFormer(**filtered_cfg)
4. load_state_dict(model.pt, strict=True)              # 这里加载的就是 EMA 权重
5. model.eval() + forward → logits → sigmoid → predictions
```

**EMA 对 infer 透明**：保存时已经 swap 到 EMA，`model.pt` 数值就是 EMA 模型，infer.py 不需要任何特殊处理。

### 4.5 当前 run.sh 配置

```bash
python3 train.py \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 4 --item_ns_tokens 2 \
    --num_queries 2 \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    --use_amp --amp_dtype bf16 \
    --use_time_features \
    --time_use_film \              # ★ FiLM
    --time_num_harmonics 1 \       # K=4 已被锁回 1
    --use_user_staleness \         # ★ 用户级 staleness
    --time_decay_pool \            # ★ decay pool
    --seq_event_mask_prob 0 \      # 失败实验已禁
    --lr_warmup_steps 500 \        # ★ cosine LR
    --lr_decay_total_steps 22000 \
    --lr_min_ratio 0.1 \
    --ema_decay 0.999 \            # ★ EMA
    --batch_size 512
```

---

## 5. 下一步候选方向

按"训练侧优化"思路，按 ROI 排序：

| 方向 | 预期收益 | 风险 | 实现成本 |
|---|---|---|---|
| **EMA decay 调参** (0.9995 / 0.9999) | +0~0.001 | 低 | 改 run.sh |
| **多 seed ensemble** (sibling 模型 logits 平均) | +0.001~0.003 | 0 | 用空卡训第二份 |
| **Cosine schedule 调参**（lr_min_ratio 0.05 等） | +0~0.001 | 低 | 改 run.sh |
| AdamW betas / weight_decay 调 | ±0.001 | 中 | 改 run.sh |

**避免再尝试**：
- 任何"加更多 time-derived signal" 的方向（已饱和）
- 增加序列层面正则化 (model 欠信号不是过拟合)
- 多阶谐波 / 给 NS pool 加更多 token

---

## 6. 关键代码引用

| 内容 | 文件:行 |
|---|---|
| `TimeFeatureEmbedder`（FiLM + staleness） | [model.py](model.py) |
| `MultiSeqQueryGenerator._pool_seq`（decay pool） | [model.py](model.py) |
| `_build_representations`（计算 user_staleness_bucket、应用 FiLM） | [model.py](model.py) |
| `PCVRHyFormerRankingTrainer.__init__`（LR scheduler + EMA 构建） | [trainer.py](trainer.py) |
| `_train_step`（lr_scheduler.step + ema.update） | [trainer.py](trainer.py) |
| `_do_validation`（EMA swap in/out） | [trainer.py](trainer.py) |
| `ModelEMA` 类 | [trainer.py](trainer.py) |
| `infer.py` 签名过滤兜底 | [infer.py](infer.py) |
