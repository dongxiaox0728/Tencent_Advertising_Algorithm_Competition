# Raw Data To Tokens In The PCVR Baseline

This note explains how the TAAC PCVR baseline transforms raw recommendation data into tensors, then into model tokens.

The key files are:

- `dataset.py`: reads raw data and converts it into PyTorch tensors.
- `train.py`: builds feature specs from the dataset schema and creates the model.
- `model.py`: converts tensors into embeddings and tokens.
- `trainer.py`: wraps each batch into `ModelInput` and calls the model.

Important: the baseline code expects a directory of `*.parquet` files plus `schema.json`. The repository also has `sample_dataset.csv`, which is useful for understanding the column layout, but `PCVRParquetDataset` does not read that CSV directly.

## 1. Raw Data Layout

Each row is one training example for PCVR, or post-click conversion rate prediction.

The raw columns are grouped into several types:

```text
ID / label columns:
  user_id
  item_id
  label_type
  label_time
  timestamp

User integer features:
  user_int_feats_1
  user_int_feats_3
  user_int_feats_4
  ...

User dense float features:
  user_dense_feats_61
  user_dense_feats_62
  ...

Item integer features:
  item_int_feats_5
  item_int_feats_6
  ...

Sequence features:
  domain_a_seq_38
  domain_a_seq_39
  ...
  domain_b_seq_67
  ...
  domain_c_seq_27
  ...
  domain_d_seq_17
  ...
```

There are two broad feature families:

```text
Non-sequence features:
  user_int_feats
  user_dense_feats
  item_int_feats
  item_dense_feats

Sequence features:
  domain_a_seq_*
  domain_b_seq_*
  domain_c_seq_*
  domain_d_seq_*
```

In this README, "token" means a learned model vector of size `d_model`, not a text token.

## 2. Schema Controls The Tensor Layout

The dataset loader reads `schema.json` in `PCVRParquetDataset._load_schema()`.

The schema tells the loader:

```text
For user_int:
  feature id
  vocab size
  fixed tensor length

For item_int:
  feature id
  vocab size
  fixed tensor length

For user_dense:
  feature id
  fixed tensor length

For each sequence domain:
  parquet column prefix
  timestamp feature id
  list of sequence feature ids and vocab sizes
```

The code uses `FeatureSchema` to remember where each feature lands inside a flattened tensor.

Example:

```text
feature 1:
  offset = 0
  length = 1

feature 15:
  offset = 3
  length = 5
```

That means:

```text
user_int_feats[:, 0]     belongs to feature 1
user_int_feats[:, 3:8]   belongs to feature 15
```

## 3. Raw Batch To Tensor Batch

The main conversion happens in:

```python
PCVRParquetDataset._convert_batch()
```

For a batch of `B` rows, the dataset returns a dictionary like:

```text
user_int_feats:       LongTensor  [B, total_user_int_dim]
item_int_feats:       LongTensor  [B, total_item_int_dim]
user_dense_feats:     FloatTensor [B, total_user_dense_dim]
item_dense_feats:     FloatTensor [B, 0]
label:                LongTensor  [B]
timestamp:            LongTensor  [B]

seq_a:                LongTensor  [B, num_seq_a_features, max_len_a]
seq_a_len:            LongTensor  [B]
seq_a_time_bucket:    LongTensor  [B, max_len_a]

seq_b:                LongTensor  [B, num_seq_b_features, max_len_b]
seq_b_len:            LongTensor  [B]
seq_b_time_bucket:    LongTensor  [B, max_len_b]

same pattern for seq_c and seq_d
```

The label is built as:

```python
label = (label_type == 2)
```

So:

```text
label_type == 2  -> label 1
otherwise        -> label 0
```

## 4. Integer Feature Tensorization

Integer features can be scalar or multi-value.

For a scalar feature:

```text
raw value:
  user_int_feats_1 = 4

tensor:
  user_int_feats[:, offset] = 4
```

For a multi-value feature:

```text
raw value:
  user_int_feats_15 = [928, 556, 538, 739, 94]
```

If the schema says `length = 8`, it becomes:

```text
[928, 556, 538, 739, 94, 0, 0, 0]
```

If the schema says `length = 3`, it becomes:

```text
[928, 556, 538]
```

The value `0` means padding or missing.

The dataset maps invalid or missing IDs to `0`:

```text
null -> 0
-1   -> 0
<=0  -> 0
```

If an ID is outside the known vocabulary range, the dataset can clip it to `0`.

## 5. Dense Feature Tensorization

Dense features are float arrays.

Example:

```text
raw:
  user_dense_feats_61 = [0.10, -0.02, 0.09, ...]

tensor:
  user_dense_feats[:, offset:offset+dim] = same float values
```

Dense features do not use embedding lookup because they are already continuous numeric values. Later, the model uses a linear projection to turn them into a token.

## 6. Sequence Tensorization

A sequence domain has several sequence feature columns.

Example for `seq_a`:

```text
domain_a_seq_38
domain_a_seq_39
domain_a_seq_40
...
```

If the configured max length is `256`, then each sequence feature is padded or truncated to length `256`.

For one user:

```text
domain_a_seq_38 = [101, 205, 88, 0, 0, ...]
domain_a_seq_39 = [7,   3,   9,  0, 0, ...]
domain_a_seq_40 = [55,  18,  2,  0, 0, ...]
```

Position `0` across all these columns describes the same historical event:

```text
Historical event 0:
  feature 38 = 101
  feature 39 = 7
  feature 40 = 55

Historical event 1:
  feature 38 = 205
  feature 39 = 3
  feature 40 = 18
```

The tensor shape is:

```text
seq_a = [B, S, L]
```

Where:

```text
B = batch size
S = number of sequence side features
L = max sequence length
```

For example:

```text
seq_a = [B, 8, 256]
```

This means:

```text
8 sequence features
256 historical positions
```

Each value inside `seq_a[b, s, l]` is an integer ID.

## 7. Timestamp And Time Buckets

Each row has a current `timestamp`. This is the time of the current target event.

Some sequence domains also have a timestamp feature for historical events. In `schema.json`, this is the domain's `ts_fid`.

For each historical event, the dataset computes:

```text
time_diff = current row timestamp - historical event timestamp
```

Example:

```text
current timestamp:
  1772725140

historical event timestamps:
  [1772725000, 1772721000, 1772600000]

time differences:
  [140 seconds, 4140 seconds, 125140 seconds]
```

The exact time differences are converted into discrete bucket IDs using `BUCKET_BOUNDARIES`.

Example idea:

```text
140 seconds ago     -> bucket 14
4140 seconds ago    -> bucket 31
125140 seconds ago  -> bucket 48
padding             -> bucket 0
```

The result is:

```text
seq_a_time_bucket = [B, L]
```

Later, the model turns these bucket IDs into time embedding vectors and adds them to the sequence event tokens.

## 8. Batch To ModelInput

The trainer converts the dataset batch dictionary into a `ModelInput` in:

```python
PCVRHyFormerRankingTrainer._make_model_input()
```

`ModelInput` contains:

```text
user_int_feats
item_int_feats
user_dense_feats
item_dense_feats
seq_data
seq_lens
seq_time_buckets
```

At this point, the data is still mostly tensors of IDs and floats. It has not become model tokens yet.

## 9. What Embedding Lookup Means

Raw integer IDs are category labels, not meaningful numeric values.

Example:

```text
ID 205 is not "larger" or "better" than ID 101.
They are just different category IDs.
```

An embedding table is a trainable lookup table:

```text
ID    learned vector
0     [0.00,  0.00,  0.00, ...]
1     [0.12, -0.03,  0.44, ...]
2     [-0.20, 0.51,  0.08, ...]
...
101   [0.34, -0.11,  0.72, ...]
205   [-0.08, 0.27,  0.19, ...]
```

When the tensor contains ID `101`, embedding lookup returns row `101` from that feature's embedding table:

```text
embedding(101) = table[101]
```

If `emb_dim = 64`, then:

```text
one integer ID -> one vector of length 64
```

The vectors start random and are learned during training.

## 10. Sequence Tensors To Sequence Tokens

Sequence token construction happens in:

```python
PCVRHyFormer._embed_seq_domain()
```

Input example:

```text
seq_a = [B, 8, 256]
emb_dim = 64
d_model = 64
```

For each sequence feature slot, the model uses a separate embedding table.

For feature slot `0`:

```text
seq_a[:, 0, :] = [B, 256] integer IDs

after embedding lookup:
  [B, 256, 64]
```

For all 8 feature slots:

```text
feature 0 embedding: [B, 256, 64]
feature 1 embedding: [B, 256, 64]
...
feature 7 embedding: [B, 256, 64]
```

Then the model concatenates the embeddings for the same event position:

```text
[B, 256, 64] concat 8 times
  -> [B, 256, 8 * 64]
  -> [B, 256, 512]
```

Why concatenate?

Because each historical event is described by multiple feature IDs. Concatenation combines all side information for the same event into one vector.

Example event:

```text
event position 0:
  feature 38 ID = 101 -> vector length 64
  feature 39 ID = 7   -> vector length 64
  feature 40 ID = 55  -> vector length 64
```

Concatenate:

```text
event 0 combined vector = [feature38_emb ; feature39_emb ; feature40_emb ; ...]
```

Then the model projects this large vector to `d_model`:

```text
Linear(S * emb_dim -> d_model)
```

For the example:

```text
Linear(512 -> 64)
```

So:

```text
[B, 256, 512] -> [B, 256, 64]
```

Finally, the time bucket embedding is added:

```text
sequence token = event feature token + time bucket embedding
```

Final sequence token shape:

```text
seq_a_tokens = [B, 256, d_model]
```

With default settings:

```text
seq_a_tokens = [B, 256, 64]
```

## 11. Non-Sequence Integer Tensors To NS Tokens

NS means non-sequence.

The integer NS tensors are:

```text
user_int_feats = [B, total_user_int_dim]
item_int_feats = [B, total_item_int_dim]
```

They are converted by either:

```text
GroupNSTokenizer
RankMixerNSTokenizer
```

The default configuration uses `RankMixerNSTokenizer`.

First, each original feature becomes one vector.

For a scalar feature:

```text
feature value:
  4

embedding lookup:
  emb(4) -> [B, emb_dim]
```

For a multi-value feature:

```text
feature values:
  [928, 556, 538, 739, 94]

embedding lookup:
  [B, length, emb_dim]

mask padding zeros and average:
  [B, emb_dim]
```

So every original sparse feature becomes:

```text
[B, emb_dim]
```

Then `RankMixerNSTokenizer` does:

```text
1. Concatenate all feature embeddings into one long vector.
2. Pad if needed.
3. Split the long vector into num_ns_tokens chunks.
4. Project each chunk to d_model.
5. Stack the projected chunks.
```

Example:

```text
20 features
emb_dim = 64
user_ns_tokens = 5
d_model = 64

after embedding:
  20 vectors, each [B, 64]

concatenate:
  [B, 20 * 64] = [B, 1280]

split into 5 chunks:
  [B, 256]
  [B, 256]
  [B, 256]
  [B, 256]
  [B, 256]

project each chunk:
  Linear(256 -> 64)

final:
  user_ns = [B, 5, 64]
```

Item integer features go through the same process:

```text
item_int_feats -> item_ns
```

## 12. Dense Features To NS Tokens

Dense features do not use embedding lookup.

For user dense features:

```text
user_dense_feats = [B, user_dense_dim]
```

The model applies:

```text
Linear(user_dense_dim -> d_model)
LayerNorm
SiLU
```

Then it adds a token dimension:

```text
user_dense_token = [B, 1, d_model]
```

This token is concatenated with the integer NS tokens.

## 13. Final NS Tokens

The model builds:

```text
user_ns
user_dense_token
item_ns
item_dense_token, if present
```

Then concatenates them:

```text
ns_tokens = [B, num_ns, d_model]
```

Example:

```text
user_ns          = [B, 5, 64]
user_dense_token = [B, 1, 64]
item_ns          = [B, 2, 64]

ns_tokens        = [B, 8, 64]
```

## 14. Query Tokens

Query tokens are generated by:

```python
MultiSeqQueryGenerator
```

They are not directly read from raw data. They are generated from:

```text
NS tokens + summarized sequence tokens
```

Input shapes:

```text
ns_tokens:      [B, num_ns, d_model]
seq_a_tokens:   [B, L_a, d_model]
seq_b_tokens:   [B, L_b, d_model]
seq_c_tokens:   [B, L_c, d_model]
seq_d_tokens:   [B, L_d, d_model]
```

For each sequence domain, the generator:

1. Flattens the NS tokens:

```text
ns_tokens [B, num_ns, d_model]
  -> ns_flat [B, num_ns * d_model]
```

2. Mean-pools the valid sequence tokens:

```text
seq_a_tokens [B, L_a, d_model]
  -> seq_a_pooled [B, d_model]
```

3. Concatenates them:

```text
global_info_a = concat(ns_flat, seq_a_pooled)
```

Shape:

```text
global_info_a = [B, (num_ns + 1) * d_model]
```

4. Feeds `global_info_a` into one or more FFNs:

```text
FFN_a_1(global_info_a) -> query token 1
FFN_a_2(global_info_a) -> query token 2
```

If `num_queries = 2`:

```text
q_a = [B, 2, d_model]
```

The model repeats this for each sequence domain:

```text
q_tokens_list = [
  q_a,
  q_b,
  q_c,
  q_d
]
```

The query token's role is to ask:

```text
Given this user/item context, what useful information should I extract from this behavior sequence?
```

Later, query tokens attend to sequence tokens through cross-attention.

## 15. Token Dimension Requirement

The model's transformer-style blocks require every token to have the same final width:

```text
d_model
```

With default settings:

```text
d_model = 64
```

Therefore all token types become:

```text
NS tokens:       [B, num_ns, 64]
sequence tokens: [B, L, 64]
query tokens:    [B, num_queries, 64]
```

This is why sequence embeddings are projected:

```text
S * emb_dim -> d_model
```

And NS chunks are also projected:

```text
chunk_dim -> d_model
```

Once everything is `[*, *, d_model]`, the HyFormer blocks can mix them with attention, residual connections, layer normalization, and RankMixer token mixing.

## 16. Full End-To-End Flow

The complete transformation is:

```text
Raw parquet row
  -> fixed-size integer and float tensors
  -> ModelInput
  -> user/item embedding lookup
  -> NS tokens
  -> sequence embedding lookup
  -> sequence event tokens
  -> query token generation
  -> HyFormer blocks
  -> classifier
  -> PCVR logit
```

More detailed:

```text
user_int_feats / item_int_feats
  -> slice by feature offset and length
  -> embedding lookup
  -> mean pool if multi-value
  -> concatenate / split / project
  -> NS tokens

user_dense_feats
  -> linear projection
  -> dense NS token

domain sequence tensors
  -> embedding lookup for each feature at each history position
  -> concatenate feature embeddings per event
  -> project to d_model
  -> add time bucket embedding
  -> sequence tokens

NS tokens + mean-pooled sequence tokens
  -> FFNs
  -> query tokens
```

The most important distinction:

```text
Tensor:
  numeric container produced by the dataset loader.

Embedding:
  learned vector for a categorical ID.

Token:
  final d_model-sized vector consumed by the model blocks.
```

