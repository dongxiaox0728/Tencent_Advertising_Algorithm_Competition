# 数据集描述总结

> 📋 本数据集为 KDD Cup 2026 竞赛 "Towards Unifying Sequence Modeling and Feature Interaction for Large-scale Recommendation" 官方发布的大规模工业级推荐数据集。

---

## 🔹 数据集概览

| 属性 | 说明 |
|------|------|
| **来源** | 腾讯广告平台真实日志（经脱敏处理） |
| **用途** | CVR（Conversion Rate）预测任务 |
| **规模** | 初步轮次包含 **2 亿** 用户行为序列 |
| **格式** | 扁平化列布局（flat column layout），无嵌套结构 |
| **隐私保护** | 所有稀疏特征为匿名整数ID，密集特征为固定长度浮点向量；**不包含**原始文本/图像/URL或个人身份信息 |

---

## 🔹 数据集组成

数据集包含两大核心组件：

```
📦 用户行为序列 (User Behavior Sequences)
├── 用户与物品的交互事件（曝光、点击、转化等）
├── 关联信息：时间戳、行为类型等

📦 非序列多字段特征 (Non-sequential Multi-field Features)
├── 用户属性（User Attributes）
├── 物品属性（Item Attributes）
├── 上下文信号（Contextual Signals）
└── 交叉特征（Cross Features）
```

---

## 🔹 列分类总览（共 120 列）

| 类别 | 列数 | 数据类型 | 描述 |
|------|------|----------|------|
| **ID & Label** | 5 | `int64` / `int32` | 核心标识符、标签、时间戳 |
| **User Int Features** | 46 | `int64` / `list<int64>` | 离散用户特征（标量 + 多值数组） |
| **User Dense Features** | 10 | `list<float>` | 连续用户特征（嵌入向量及对齐统计量） |
| **Item Int Features** | 14 | `int64` / `list<int64>` | 离散物品特征（类别、类型、多标签等） |
| **Domain Sequence Features** | 45 | `list<int64>` | 4 个行为域的用户序列特征 |

---

## 🔹 详细列说明

### 1️⃣ ID & Label 列（5 列，无空值）

| 列名 | 数据类型 | 说明 |
|------|----------|------|
| `user_id` | `int64` | 用户唯一标识 |
| `item_id` | `int64` | 物品唯一标识 |
| `label_type` | `int32` | 标签类型 |
| `label_time` | `int64` | 标签发生时间 |
| `timestamp` | `int64` | 事件时间戳 |

---

### 2️⃣ User Int Features（46 列）

```markdown
🔹 标量特征（Scalar, int64）- 35 列
   user_int_feats_{1, 3, 4, 48-59, 82, 86, 92-109}

🔹 数组特征（Array, list<int64>）- 11 列
   user_int_feats_{15, 60, 62-66, 80, 89-91}
```

> 📌 特征内容：年龄、性别、婚姻状况等用户基本属性与偏好

---

### 3️⃣ User Dense Features（10 列，`list<float>`）

| 特征组 | 列索引 | 说明 |
|--------|--------|------|
| 用户嵌入向量 | `61, 87` | SUM、LMF4Ads 等嵌入特征 |
| 对齐统计特征 | `62-66, 89-91` | 与对应 `user_int_feats` **元素级对齐**的连续值 |

#### 🔗 对齐示例
```python
# 假设：
user_int_feats_62:   [1, 2, 3]           # 离散ID
user_dense_feats_62: [10.5, 20.0, 15.5]  # 对应统计量

# 含义：
# ID=1 → 统计值10.5（如停留时长/得分/概率）
# ID=2 → 统计值20.0
# ID=3 → 统计值15.5
```

---

### 4️⃣ Item Int Features（14 列）

```markdown
🔹 标量特征（Scalar, int64）- 13 列
   item_int_feats_{5-10, 12-13, 16, 81, 83-85}

🔹 数组特征（Array, list<int64>）- 1 列
   item_int_feats_{11}
```

> 📌 特征内容：物品类别、类型、基础信息及多标签标注

---

### 5️⃣ Domain Sequence Features（45 列，`list<int64>`）

用户行为序列按 **4 个行为域** 划分：

| 域 | 列索引 | 列数 |
|----|--------|------|
| **Domain A** | `38-46` | 9 |
| **Domain B** | `67-79, 88` | 14 |
| **Domain C** | `27-37, 47` | 12 |
| **Domain D** | `17-26` | 10 |

> 📌 每列为一个行为序列（如点击序列、曝光序列等），元素为匿名物品ID

---

## 🔹 使用注意事项

```⚠️
1. 所有特征均为匿名化处理，不可逆推原始信息
2. 密集特征与离散特征存在对齐关系，使用时需注意元素级对应
3. 序列特征为变长列表，需采用适当的方式进行处理（如padding、attention等）
4. 评估指标为 ROC-AUC，同时需满足推理延迟约束
```

---

> 📬 数据集问题咨询：`taac@tencent.com`  
> 🏢 版权归属：© 1998-2026 Tencent. All Rights Reserved.

---

*最后更新：2026-04-25 | 竞赛官网：algo.qq.com*