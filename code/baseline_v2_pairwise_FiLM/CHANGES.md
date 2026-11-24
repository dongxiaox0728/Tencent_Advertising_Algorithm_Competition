# baseline_v2_pairwise_FiLM — Changes

基于 `baseline_amp_FiLM_decay_pool_staleness_cosine_lr22k_ema999`，保留所有原有模块（AMP/EMA/Cosine LR/Time FiLM/DecayPool/Staleness），新增 **Pair-level FiLM** 特征工程。

## 核心改动

### 新增：PairwiseFiLMPooler（model.py）

对 fid 62-66 和 89-91 这 8 个 int-dense 对齐字段，在池化时用 dense stat 做 FiLM 调制：

```
norm_stat = sign(x) * log1p(|x|)
gamma, beta = Linear(norm_stat)           # 零初始化
pair_emb = id_emb * (1 + gamma) + beta   # FiLM 调制
repr = masked_mean_pool(pair_emb)
```

原理：dense stat 反映用户与该实体的交互强度，通过乘性调制让每个 id_emb 的各维度被差异化地放大/压缩，比原来的加性 residual MLP 表达力更强。

### 修改：RankMixerNSTokenizer（model.py）

- `__init__` 新增 `paired_fid_info` 参数，为每个 pairwise fid 创建独立的 `PairwiseFiLMPooler`
- `forward` 新增 `user_dense_feats` 参数；对 pairwise fid 用 FiLM pooler 替代等权均值池化

### 修改：train.py

启动时自动从 `schema.json` 检测 pairwise fid（int 和 dense schema 中 fid 相同且 dim > 1 且 length 匹配），构建 `paired_user_fid_info` 传入模型。无需手动配置。

## 不变的部分

- `dataset.py`：无改动，user_dense_feats 仍作为整体 flat tensor 输出
- `trainer.py`：无改动，user_dense_feats 已通过 ModelInput 传入模型
- `user_dense_proj`：保留，作为 dense 特征的并行全局路径（与 pair FiLM 互补）
- Token 数量：不变（T=16, d_model=64, 64%16=0 ✓）
- 所有超参数：与 baseline 版本相同
