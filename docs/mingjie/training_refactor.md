# 训练代码改装计划：BF16 混合精度集成

> 目标：在不破坏现有双优化器架构与 checkpoint 兼容性的前提下，将 BF16 自动混合精度（AMP）接入训练循环，预期吞吐 +50~100%、显存峰值 -25~35%。

---

## 1. 背景与判定依据

### 1.1 当前训练剖面（实测）
| 指标 | 数值 | 解读 |
|---|---|---|
| GPU 利用率 | ~100% | 算力满载，IO 不是瓶颈 |
| GPU 显存 | >90% | 显存吃紧，离 OOM 一步之遥 |
| CPU 利用率 | ~13% | DataLoader 流水线游刃有余 |

**判定**：纯算力 + 显存瓶颈。AMP 是收益最直接、改动最小的优化路径。DataLoader / IO 相关优化暂不做。

### 1.2 项目特点（影响方案选择）
- **双优化器**：Adagrad（稀疏 Embedding）+ AdamW（dense 参数），见 [trainer.py:84-94](../../code/TAAC_baseline/trainer.py#L84-L94)。
- **梯度裁剪已存在**：`clip_grad_norm_(max_norm=1.0, foreach=False)`，`foreach=False` 是绕开 PyTorch 一个 CUDA kernel bug。
- **历史上出现过 NaN logits**：`evaluate()` 已经在过滤 NaN（见 [trainer.py:460-466](../../code/TAAC_baseline/trainer.py#L460-L466)），说明梯度爆炸有先例。
- **稀疏梯度路径**：Embedding 在 Adagrad sparse 模式下可能产生 `torch.sparse` 张量，这条路径与 `GradScaler` 的交互需要警惕。
- **四个版本共享同一 trainer 结构**：TAAC_baseline / UniRec_V1 / V2 / V3 的 `train.py` 与 `trainer.py` 仅命令行参数与默认值差异，核心循环一致。

### 1.3 精度选型决策
**采用 BF16，不采用 FP16。** 理由：
1. 项目有梯度爆炸史，FP16 窄动态范围会放大风险；BF16 指数位与 FP32 对齐，动态范围安全。
2. BF16 不需要 `GradScaler`，避免与双优化器、稀疏梯度的复杂耦合。
3. 假设训练硬件为 Ampere 或更新（A100 / A10 / RTX 30/40 系列）。**前置校验**：`torch.cuda.is_bf16_supported()` 必须为 `True`，否则降级到 FP16 + GradScaler 路径（保留代码兼容性，但默认禁用）。

---

## 2. 改动范围

只动两个文件，每个版本各生成一份新的（基线先做，其他版本机械同步）：
- [`code/TAAC_baseline/trainer.py`](../../code/TAAC_baseline/trainer.py) ==>> code/train_amp/trainer.py
- [`code/TAAC_baseline/train.py`](../../code/TAAC_baseline/train.py) ==>> code/train_amp/train.py

**不动**：`model.py`、`dataset.py`、`utils.py`、`run.sh`、checkpoint 格式与 sidecar 文件。

---

## 3. 详细改装步骤

### Step 1：`train.py` — 新增命令行开关
在 `parse_args()` 末尾、环境变量覆盖之前，新增：

```python
# Mixed precision.
parser.add_argument('--use_amp', action='store_true', default=True,
                    help='Enable automatic mixed precision (default on)')
parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                    help='Disable AMP, fall back to FP32')
parser.add_argument('--amp_dtype', type=str, default='bf16',
                    choices=['bf16', 'fp16'],
                    help='AMP dtype: bf16 (default, Ampere+) or fp16 (older GPUs, '
                         'enables GradScaler)')
```

在 `main()` 构造 trainer 时透传：
```python
trainer = PCVRHyFormerRankingTrainer(
    ...,
    use_amp=args.use_amp,
    amp_dtype=args.amp_dtype,
    ...,
)
```

**记得把 `train_config = vars(args)` 自动包含这两个新字段** —— sidecar `train_config.json` 会自动序列化，infer 时可读取（如 infer 端也接 AMP）。

### Step 2：`trainer.py` — `__init__` 增加 AMP 初始化
新增构造参数 `use_amp: bool = True`、`amp_dtype: str = 'bf16'`。

构造体内逻辑：
```python
self.use_amp = use_amp
self.amp_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}[amp_dtype]

# BF16 不需要 GradScaler；FP16 才需要。
if use_amp and amp_dtype == 'fp16':
    self.scaler = torch.amp.GradScaler('cuda')
else:
    self.scaler = None

# 启动期硬件校验，失败直接退化到 FP32 + 警告，避免悄悄跑错。
if use_amp and amp_dtype == 'bf16' and not torch.cuda.is_bf16_supported():
    logging.warning("BF16 not supported on this GPU, falling back to FP32")
    self.use_amp = False

logging.info(f"AMP: use_amp={self.use_amp}, amp_dtype={amp_dtype}, "
             f"scaler={'on' if self.scaler else 'off'}")
```

### Step 3：`trainer.py` — `_train_step` 重写
关键点：
1. 前向 + 损失计算包在 `autocast` 内。
2. BF16 路径：`backward → clip → step` 与原代码一致。
3. FP16 路径：`scale → backward → unscale_(每个 optimizer) → clip → scaler.step(每个 optimizer) → scaler.update()`。
4. 梯度裁剪保留 `foreach=False`。

```python
def _train_step(self, batch: Dict[str, Any]) -> float:
    device_batch = self._batch_to_device(batch)
    label = device_batch['label'].float()

    self.dense_optimizer.zero_grad()
    if self.sparse_optimizer is not None:
        self.sparse_optimizer.zero_grad()

    model_input = self._make_model_input(device_batch)

    with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
        logits = self.model(model_input).squeeze(-1)
        if self.loss_type == 'focal':
            loss = sigmoid_focal_loss(logits, label,
                                      alpha=self.focal_alpha,
                                      gamma=self.focal_gamma)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, label)

    if self.scaler is not None:
        # FP16 path
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.dense_optimizer)
        if self.sparse_optimizer is not None:
            self.scaler.unscale_(self.sparse_optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0, foreach=False)
        self.scaler.step(self.dense_optimizer)
        if self.sparse_optimizer is not None:
            self.scaler.step(self.sparse_optimizer)
        self.scaler.update()
    else:
        # BF16 / FP32 path (no scaler)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0, foreach=False)
        self.dense_optimizer.step()
        if self.sparse_optimizer is not None:
            self.sparse_optimizer.step()

    return loss.item()
```

### Step 4：`trainer.py` — `_evaluate_step` 包 autocast
推理无反传，**不要碰 scaler**。logits 出 autocast 后立刻 `.float()` 转回 FP32，避免 AUC / logloss 计算精度损失：

```python
def _evaluate_step(self, batch):
    device_batch = self._batch_to_device(batch)
    label = device_batch['label']
    model_input = self._make_model_input(device_batch)
    with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
        logits, _ = self.model.predict(model_invocation_input := model_input)
    logits = logits.squeeze(-1).float()
    return logits, label
```

### Step 5：稀疏重置路径校验
[trainer.py:355-376](../../code/TAAC_baseline/trainer.py#L355-L376) 的 Adagrad 重建逻辑**不需要改动** —— 它只动 optimizer state，不依赖梯度精度。但要在第一次跑时观察日志，确认 `restored` 计数与未启用 AMP 时一致。

---

## 4. 验证计划

### 4.1 正确性验证（必须做，按顺序）
| 步骤 | 操作 | 通过标准 |
|---|---|---|
| 1 | 在小数据子集（`--train_ratio 0.05`）跑 1 epoch FP32，记录 train loss 曲线、val AUC | 作为 baseline 对照组 |
| 2 | 同样配置跑 BF16 AMP 1 epoch | train loss 曲线相对偏差 <2%，val AUC 偏差 <0.001 |
| 3 | 检查日志中 NaN logits 计数 | AMP 下不应显著高于 FP32（理想为 0） |
| 4 | 加载 AMP checkpoint 用 `infer.py` 跑一次 | 输出 shape / 数值范围正常 |

### 4.2 性能验证
- 全量数据跑 1 epoch，对比：
  - **吞吐**：`tqdm` 显示的 `it/s`，预期 +50~100%。
  - **显存峰值**：`nvidia-smi` 或 `torch.cuda.max_memory_allocated()`，预期 -25~35%。
- 若显存有富余，尝试 `batch_size 256 → 384 / 512`，记录 AUC 是否进一步提升（学习率按 √(new/old) 略调）。

### 4.3 回归保险
- 保留 `--no_amp` 开关，任何时候都能一键回退到 FP32。
- 保留 FP16 + GradScaler 路径，便于在老 GPU 上调试时切换。

---

## 5. 风险与对策

| 风险 | 概率 | 应对 |
|---|---|---|
| 首次开 AMP 时 BF16 cast 缓冲触发瞬时 OOM（显存已 >90%） | 中 | 第一次跑前把 batch_size 临时调小 25%，确认稳定再调回 |
| BF16 精度下 attention softmax 数值不稳，AUC 退化 | 低 | autocast 默认会把 softmax / layernorm 留在 FP32，无需手动处理；若仍出问题，用 `with autocast(enabled=False):` 局部包住敏感算子 |
| 稀疏 Adagrad 与 AMP 交互异常 | 低 | BF16 路径完全不碰 scaler，稀疏路径与原代码完全一致；只在 FP16 路径需要小心 |
| `model.predict()` 返回的中间表征精度被 BF16 污染下游使用 | 低 | 评估只用 logits（已 `.float()`）；如果 infer 端用到了第二个返回值，需在 infer 端同样 `.float()` |

---

## 6. 推进顺序

1. **本地小数据验证**（1~2 小时）：先在 `TAAC_baseline` 上做 Step 1~4 改动，跑 4.1 的正确性验证。
2. **基线全量训练**（1 天）：通过后跑全量，记录吞吐 / 显存 / 收敛曲线，与历史 FP32 run 对比。
3. **同步到三个版本**：把改动机械地复制到 `UniRec_V1 / V2 / V3` 的 `trainer.py` 与 `train.py`（这两个文件在四个版本中差异极小，diff-patch 即可）。
4. **可选后续优化**（视收益再决定）：
   - 大 batch_size 试验（吃掉 AMP 省下的显存）。
   - `torch.compile(model, mode='reduce-overhead')`，HyFormer 类结构通常再 +10~20%。
   - DataLoader 调参 **暂不做**（CPU 仅 13%，无空间）。

---

## 7. 不做的事（明确列出，避免范围蔓延）

- **不重构** trainer 类结构、不抽象 AMP wrapper。
- **不改** dataset / model / utils。
- **不改动** checkpoint 文件格式与 sidecar 文件。
- **不引入** Apex、DeepSpeed、Accelerate 等外部框架。保持纯 PyTorch AMP，依赖最小。
- **不优化** DataLoader（CPU 13%，无瓶颈）。
- **不动** `loss_type` / `pos_weight` / `focal_alpha` 等损失相关超参（与 AMP 正交）。