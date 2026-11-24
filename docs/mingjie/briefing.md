# Complete

# Update

## weighted-BCE vs Focal Loss：PCVR 里为什么要选 Focal Loss？

PCVR 是极不均衡的二分类，负样本可以是正样本的几十倍，所以才会有 Focal Loss 这种针对性设计

shell 里有个选项，可以选择 BCE loss or Focal Loss。 

Apr26 的第一次baseline只用了BCE。可以考虑明天的第二次baseline改成Focal Loss，看看效果。


## UniRec V1 去掉了 Hyformer 的 Token Mixer/MLP Mixer

二、评价：UniRECBlock 去掉 Token Mixing 的影响
UniRECBlock 完全删除了这个步骤（rank_mixer_mode 标注 unused）。下面分四个层面评估影响。
2.1 影响一：失去了"NS 与所有 Q 之间"的轻量交互通道
这是最直接的功能损失。在原 HyFormer 里，Token Mixing 让 16 个 token（Q + NS）全部互相洗牌，包括：

✅ Q_seq1 ↔ Q_seq2 ↔ Q_seq3（不同序列 Q 之间的交互）
✅ Q ↔ NS（Q 和非序列特征之间的交互）
✅ NS ↔ NS（不同 NS token 之间的交互）

UniRECBlock 通过双向 cross-attention 显式实现了 NS↔S 的交互，但漏掉了几个：
交互对HyFormer 用什么实现UniRECBlock 用什么实现是否覆盖NS ↔ SBoosting（间接）NS↔S 双向 cross-attn✅ 覆盖（且更强）Q ↔ SQuery DecodingQ←S cross-attn✅ 覆盖NS ↔ NSToken Mixing无（每个 NS 独立 FFN）❌ 丢失Q ↔ Q（跨序列）Token Mixing无❌ 丢失Q ↔ NSToken Mixing无（间接通过 S 中转）❌ 丢失
这是 UniRECBlock 的一个潜在盲点：它特别擅长 NS↔S 这一对，但牺牲了 token 之间的"全连通"性质。
2.2 影响二：跨序列 Q 的交互被弱化
回想 HyFormer 处理多序列的设计哲学（论文 3.7 节）：

"preserves sequence-specific semantics during decoding, while enabling cross-sequence interaction to be handled later through query-level token mixing"

Token Mixing 在原架构里承担了"跨序列汇合"的职责——3 条序列各自的 Q 在 Boosting 时被混合。UniRECBlock 删掉这一步后：

Q_seq1、Q_seq2、Q_seq3 各自独立查询自己的序列
它们之间在 Block 内没有任何直接交互
唯一的间接路径：通过 NS（NS←S 时吸收所有序列的信息，再通过 NS→S 反馈到序列）

这条间接路径绕了一大圈，而且 NS 在 NS←S 时是对 S 个序列做平均（ns_delta.mean(dim=0)），细节信息会被平均掉。跨序列的精细交互能力变弱了。
2.3 影响三：NS 之间的"全局协调"丢失
UniRECBlock 给每个 NS token 配独立的 FFN（参数不共享），这是好处也是代价：

好处：每个 NS slot 学到专属的语义角色（user 画像 vs item 画像）
代价：NS 之间彼此完全孤立，没有任何交互

举例：假设有两个 NS token，"用户兴趣画像"和"上下文信息"。原 HyFormer 在 Token Mixing 时它们会互相洗牌，自然形成"用户兴趣 × 上下文"的隐式特征交叉。UniRECBlock 里它们各走各的 FFN，这种交叉只能通过 S（序列）做中转——非常间接。
如果业务上 NS 内部的特征交叉很重要（比如老的 DCN/DeepFM 强调的"feature crossing"），UniRECBlock 在这一点上是退化的。
2.4 影响四：算力收支重新核算
虽然删除 Token Mixing 看起来省了算力，但 UniRECBlock 加了双向 cross-attn，整体反而贵很多。粗略估算每层 Block：
步骤HyFormerUniRECBlockSequence Encoder~ X~ XQ ← S cross-attn1 次1 次NS ← S cross-attn-+ S 次S ← NS cross-attn-+ S 次Token Mixing0（免费）-Per-token FFN1 次（共享）Nns 次（独立）整体相对成本基准大约 2-3 倍
原 HyFormer 用"免费的 Token Mixing"换得了 token 间全连通；UniRECBlock 用"昂贵的 cross-attn"换得了精确的 NS↔S 交互，但通用性反而降低了。
2.5 论文消融的参考数据
论文 Table 2 里有个直接相关的数字：
配置AUC损失HyFormer 完整0.6489-HyFormer w/o Global Tokens0.6484-0.08%
这里 "w/o Global Tokens" 大致等价于"去掉了 Boosting 的核心混合功能"。0.08% 的损失说明 Token Mixing + Per-token FFN 的整体贡献是实打实的。UniRECBlock 删掉这套机制，得用双向 cross-attn 至少补回这 0.08%，否则就是净亏损。

## 特征工程

用NonSeq+Seq拼出来新的特征，放进整条流水线

## 双卡支持

我不太会。研究一下taiji平台。

# 在Sequtional tokenizer之后，对 Seq_token 添加 RoPE 位置编码

改一个参数 `use_rope=True`

# 试一下semantic group tokenizer
改shell 参数

# 试一下混合 semanticGroup + RankMixerNSTokenizer
在一个sematic group里做 rankmixer类的Tokenizer，看看效果

在model中添加一个tokenizer，添加参数选项，改shell参数

# 为不同sequence生成不同数量的global token
对于重要的seq，让他和NS 拼出来的feature，FFN成更多的global token。对于不重要的seq，生成更少的global token。
