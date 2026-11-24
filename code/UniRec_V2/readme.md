# UniRec V2 · hash embeddings for long-tail IDs

**Comparison baseline** · [UniRec V1](../UniRec_V1/) | [Project version guide](../../readme.md).

## Design and motivation

V2 handles each high-cardinality sequence field by allocating a field-specific table with 500,000 hash buckets and a separate padding row. It maps observed IDs by modulo and remaps a nonpadding remainder of zero to one. This lets fields above the vocabulary threshold contribute to learning without storing the full vocabulary. The original notes identify one affected field in `seq_b` and three in `seq_c` at the one-million threshold. Hash collisions trade fidelity for bounded memory. The [AMP hash experiment](../baseline_amp_hash_embedding/) instead shares one table across fields.

`_make_seq_embs` records which tables use hashing and `_embed_seq_domain` applies the remapping before lookup. The training log reports hash-embedded fields by domain.

## Running this snapshot

Use `python3 train.py --help` to inspect the options. Set `--emb_skip_threshold 1000000` for the recorded threshold and select the intended `--block_type` explicitly. This folder has no launcher. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

**0.8130 online AUC** is attributed to UniRec V2 in the [experiment history](../baseline_film_decay_staleness_cosine_film_on_seq/#cumulative-experiment-chain-how-we-got-here). The local notes do not identify the exact run arguments.
