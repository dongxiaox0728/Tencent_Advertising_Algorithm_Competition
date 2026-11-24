# UniRec V3 · parallel interaction and target-query prediction

**Comparison baseline** · [UniRec V2](../UniRec_V2/) | [Project version guide](../../readme.md).

## Design and motivation

V3 carries forward sparse history handling, hash embeddings, and the class-weighted loss options. Its two architecture changes make the model's readout more directly dependent on the enriched static features and target ad.

1. **Parallel query readout.** The bidirectional cross-attention updates use the pre-update sequence and NS features as inputs. Domain queries then read the concatenation of the enriched sequence and enriched NS tokens. The padding mask is extended with unmasked positions for NS, and this mixed-key attention step omits RoPE because static tokens have no sequence position. Serial mode reads only the enriched sequence at that step.
2. **Target-query head.** Mean pooling of final item NS tokens creates one candidate-ad query. This query attends over the concatenated domain queries, and the resulting vector is normalized and classified. The original flat head directly projects the concatenated queries, which can already contain item information through NS conditioning. The new head gives the target ad an explicit role in the final readout.

## Running this snapshot

Use `python3 train.py --block_type unirec` with the path and tokenizer arguments for the intended run. This snapshot has no launcher. Training parses `--interaction_mode` and `--use_target_query_head` but does not pass either into model construction, so the constructor uses `parallel` and `True`. Inference reads training configuration and falls back to `serial` and `False` when the corresponding settings are unavailable. Verify the effective architecture when loading older checkpoints. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

No final evaluation score is recorded for this snapshot.
