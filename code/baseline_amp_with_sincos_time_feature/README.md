# Sinusoidal time-conditioning and ablation suite

**Comparison baseline** · [the sequence-FiLM control](../baseline_film_decay_staleness_cosine_film_on_seq/) | [Project version guide](../../readme.md).

## Design and motivation

This snapshot keeps the sinusoidal calendar token, global staleness, decay pooling, and sequence FiLM. It collects alternative launchers to test where time should enter the model and whether domain histories should exchange information.

| Launcher | Purpose |
| --- | --- |
| `run.sh` | Sequence FiLM with learning rate `3e-4`, patience 10, a 30k decay schedule, and EMA off. |
| `run_film_on_seq.sh` | Sequence-FiLM configuration for comparison. |
| `run_film_on_q.sh` | Also applies the request-time scale and shift to generated query tokens. |
| `run_cross_domain.sh` | Lets a domain query read other behavior domains as an additional source. |
| `run_per_domain_staleness.sh` | Tests separate most-recent-event signals for each domain. |
| `run_activity_count.sh` | Tests bucketed per-domain event counts. |

Compare the launch flags before attributing a result to one mechanism. Extending FiLM to queries tests whether time affects what the model retrieves, beyond its effect on user, ad, and event representations.

## Running this snapshot

Run a launcher from the table with `bash`. The [historical iteration notes](ITERATION_NOTES.md) describe early time experiments, and the [English EMA branch README](../baseline_amp_FiLM_decay_pool_staleness_cosine_lr22k_ema999/) gives the main milestones. See the [data and environment setup](../../readme.md#running-a-snapshot).

## Results

The historical query-FiLM run recorded **0.8213 online AUC**. The sequence-FiLM control recorded **0.823064**. The records do not identify a final score for this folder’s current default launcher or the cross-domain experiment. [Sequence-FiLM run notes](../baseline_film_decay_staleness_cosine_film_on_seq/).
