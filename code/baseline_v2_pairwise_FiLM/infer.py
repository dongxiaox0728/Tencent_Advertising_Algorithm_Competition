"""PCVRHyFormer inference script for baseline_time_periodic.

Model construction mirrors ``train.py``. The checkpoint directory should
contain ``model.pt`` plus the sidecar files written by ``trainer.py``:
``schema.json``, optional ``ns_groups.json``, and ``train_config.json``.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory containing ``model.pt``, or a
                       parent directory containing ``global_step*`` checkpoint
                       subdirectories.
    EVAL_DATA_PATH     Test data directory or a single test parquet file.
    EVAL_RESULT_PATH   Directory for the generated ``predictions.json``.
"""

import inspect
import json
import logging
import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FeatureSchema, NUM_TIME_BUCKETS, PCVRParquetDataset
from model import ModelInput, PCVRHyFormer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Used only when train_config.json is absent. These values match this folder's
# active run.sh profile and train.py defaults.
_FALLBACK_MODEL_CFG = {
    "d_model": 64,
    "emb_dim": 64,
    "num_queries": 2,
    "num_hyformer_blocks": 2,
    "num_heads": 4,
    "seq_encoder_type": "transformer",
    "hidden_mult": 4,
    "dropout_rate": 0.01,
    "seq_top_k": 50,
    "seq_causal": False,
    "action_num": 1,
    "num_time_buckets": NUM_TIME_BUCKETS,
    "rank_mixer_mode": "full",
    "use_rope": False,
    "rope_base": 10000.0,
    "emb_skip_threshold": 1000000,
    "seq_id_threshold": 10000,
    "ns_tokenizer_type": "rankmixer",
    "user_ns_tokens": 4,
    "item_ns_tokens": 2,
    "use_time_features": True,
    "time_emb_dim": 16,
    "time_use_film": True,
    "time_num_harmonics": 1,
    "use_user_staleness": True,
    "staleness_mode": "global",
    "use_activity_count": False,
    "num_activity_buckets": 10,
    "time_decay_pool": True,
    "seq_event_mask_prob": 0.0,
}

_FALLBACK_SEQ_MAX_LENS = "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
_FALLBACK_BATCH_SIZE = 512
_FALLBACK_NUM_WORKERS = 8
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build ``[(vocab_size, offset, length), ...]`` in schema entry order."""
    specs: List[Tuple[int, int, int]] = []
    for _fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    seq_max_lens: Dict[str, int] = {}
    if not sml_str:
        return seq_max_lens
    for pair in sml_str.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split(":")
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"Required environment variable {name} is not set")
    return value


def _prediction_key(user_id: Any) -> str:
    return str(user_id)


def _can_start_worker_processes() -> bool:
    try:
        queue = mp.get_context().Queue()
        queue.close()
        queue.join_thread()
        return True
    except (OSError, PermissionError):
        return False


def resolve_model_dir(path: str) -> str:
    """Return the concrete checkpoint directory containing ``model.pt``."""
    if os.path.isfile(os.path.join(path, "model.pt")):
        return path

    candidates = []
    if os.path.isdir(path):
        for name in os.listdir(path):
            child = os.path.join(path, name)
            if os.path.isdir(child) and os.path.isfile(os.path.join(child, "model.pt")):
                candidates.append(child)

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint directory containing model.pt found under {path!r}"
        )

    best = [c for c in candidates if c.endswith(".best_model")]
    selected = sorted(best or candidates, key=os.path.getmtime)[-1]
    logging.info(f"Resolved checkpoint directory: {selected}")
    return selected


def load_train_config(model_dir: str) -> Dict[str, Any]:
    train_config_path = os.path.join(model_dir, "train_config.json")
    if os.path.exists(train_config_path):
        with open(train_config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg

    logging.warning(
        f"train_config.json not found in {model_dir}; falling back to hardcoded "
        "defaults. Shape mismatch may occur if training used non-default "
        "hyperparameters."
    )
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == "num_time_buckets":
            if "num_time_buckets" in train_config:
                cfg[key] = train_config["num_time_buckets"]
            elif "use_time_buckets" in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config["use_time_buckets"] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
                logging.warning(
                    "train_config missing both 'num_time_buckets' and "
                    f"'use_time_buckets', using fallback = {cfg[key]}"
                )
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(f"train_config missing '{key}', using fallback = {cfg[key]}")
    return cfg


def _load_ns_groups(
    dataset: PCVRParquetDataset,
    ns_groups_json: Optional[str],
) -> Tuple[List[List[int]], List[List[int]]]:
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, "r", encoding="utf-8") as f:
            ns_groups_cfg = json.load(f)

        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        try:
            user_ns_groups = [
                [user_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg["user_ns_groups"].values()
            ]
            item_ns_groups = [
                [item_fid_to_idx[f] for f in fids]
                for fids in ns_groups_cfg["item_ns_groups"].values()
            ]
        except KeyError as exc:
            raise KeyError(
                f"ns_groups.json references fid {exc.args[0]} that is not "
                "present in schema.json. Use the sidecar files from the same "
                "training checkpoint."
            ) from exc
        return user_ns_groups, item_ns_groups

    logging.info("No NS groups JSON found, using default singleton groups")
    return (
        [[i] for i in range(len(dataset.user_int_schema.entries))],
        [[i] for i in range(len(dataset.item_int_schema.entries))],
    )


def _detect_paired_user_fid_info(
    dataset: PCVRParquetDataset,
) -> List[Tuple[int, int, int]]:
    # Mirrors train.py: detect (id, stat) aligned pairs where the same fid
    # exists in both user_int and user_dense schemas with matching length>1.
    # Must stay byte-equivalent to the training-side detection, otherwise the
    # checkpoint's pair_film_modules.* keys won't match.
    int_fid_map = {
        fid: (idx, offset, length)
        for idx, (fid, offset, length) in enumerate(dataset.user_int_schema.entries)
    }
    dense_fid_map = {
        fid: (offset, length)
        for fid, offset, length in dataset.user_dense_schema.entries
    }
    paired: List[Tuple[int, int, int]] = []
    for fid, (int_idx, _int_offset, int_length) in int_fid_map.items():
        if fid in dense_fid_map:
            dense_offset, dense_length = dense_fid_map[fid]
            if dense_length == int_length and int_length > 1:
                paired.append((int_idx, dense_offset, dense_length))
    return paired


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str],
    device: str,
) -> PCVRHyFormer:
    user_ns_groups, item_ns_groups = _load_ns_groups(dataset, ns_groups_json)
    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema,
        dataset.user_int_vocab_sizes,
    )
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema,
        dataset.item_int_vocab_sizes,
    )

    paired_user_fid_info = _detect_paired_user_fid_info(dataset)
    logging.info(
        f"Pairwise user fids detected: {len(paired_user_fid_info)} "
        f"(int_idx list: {[p[0] for p in paired_user_fid_info]})"
    )

    # Filter model_cfg by the actual PCVRHyFormer.__init__ signature so that
    # newer config keys (added in later code revisions) don't blow up when the
    # inference-time model.py is older (e.g. bundled with an older training
    # job). Training-only kwargs like seq_event_mask_prob become no-ops here.
    accepted = set(inspect.signature(PCVRHyFormer.__init__).parameters.keys())
    filtered_cfg = {k: v for k, v in model_cfg.items() if k in accepted}
    dropped = sorted(set(model_cfg) - set(filtered_cfg))
    if dropped:
        logging.warning(
            f"Dropping cfg keys not accepted by current PCVRHyFormer.__init__: {dropped}")

    extra_kwargs: Dict[str, Any] = {}
    if "paired_user_fid_info" in accepted:
        extra_kwargs["paired_user_fid_info"] = paired_user_fid_info
    elif paired_user_fid_info:
        raise RuntimeError(
            f"Detected {len(paired_user_fid_info)} pairwise user fids in schema, "
            "but the current model.py's PCVRHyFormer.__init__ does not accept "
            "'paired_user_fid_info'. The checkpoint contains pair_film_modules.* "
            "weights that cannot be loaded. Use a model.py that supports pair-FiLM."
        )

    logging.info(f"Building PCVRHyFormer with cfg: {filtered_cfg}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **extra_kwargs,
        **filtered_cfg,
    ).to(device)
    return model


def load_model_state_strict(model: nn.Module, ckpt_path: str, device: str) -> None:
    state_dict = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        logging.error(
            "Failed to load state_dict in strict mode. This usually means "
            "infer.py rebuilt a model shape that differs from the checkpoint. "
            "Check schema.json, ns_groups.json, and train_config.json."
        )
        raise exc


def _batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v
    return device_batch


def _batch_to_model_input(device_batch: Dict[str, Any], device: str) -> ModelInput:
    seq_domains = device_batch["_seq_domains"]
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f"{domain}_len"]
        bsz = device_batch[domain].shape[0]
        seq_len = device_batch[domain].shape[2]
        seq_time_buckets[domain] = device_batch.get(
            f"{domain}_time_bucket",
            torch.zeros(bsz, seq_len, dtype=torch.long, device=device),
        )

    return ModelInput(
        user_int_feats=device_batch["user_int_feats"],
        item_int_feats=device_batch["item_int_feats"],
        user_dense_feats=device_batch["user_dense_feats"],
        item_dense_feats=device_batch["item_dense_feats"],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        time_hour=device_batch.get("time_hour"),
        time_day=device_batch.get("time_day"),
        time_hour_of_week=device_batch.get("time_hour_of_week"),
    )


def _make_data_loader(
    test_dataset: PCVRParquetDataset,
    num_workers: int,
) -> DataLoader:
    loader_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )


def main() -> None:
    model_dir = resolve_model_dir(_require_env("MODEL_OUTPUT_PATH"))
    data_dir = _require_env("EVAL_DATA_PATH")
    result_dir = _require_env("EVAL_RESULT_PATH")
    os.makedirs(result_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    schema_path = os.path.join(model_dir, "schema.json")
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, "schema.json")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema.json not found in {model_dir!r} or {data_dir!r}")
    logging.info(f"Using schema: {schema_path}")

    train_config = load_train_config(model_dir)
    seq_max_lens = _parse_seq_max_lens(
        train_config.get("seq_max_lens", _FALLBACK_SEQ_MAX_LENS)
    )
    logging.info(f"seq_max_lens: {seq_max_lens}")

    batch_size = int(train_config.get("batch_size", _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get("num_workers", _FALLBACK_NUM_WORKERS))
    if num_workers > 0 and not _can_start_worker_processes():
        logging.warning(
            "DataLoader workers are unavailable in this runtime; using "
            "num_workers=0 for inference."
        )
        num_workers = 0
    logging.info(
        "Inference data settings: batch_size=%d, num_workers=%d",
        batch_size,
        num_workers,
    )

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )
    logging.info(f"Total test samples: {test_dataset.num_rows}")

    ns_groups_json = train_config.get("ns_groups_json")
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model_cfg = resolve_model_cfg(train_config)
    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    ckpt_path = os.path.join(model_dir, "model.pt")
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    def collect_predictions(loader: DataLoader) -> Dict[str, float]:
        predictions: Dict[str, float] = {}
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                device_batch = _batch_to_device(batch, device)
                model_input = _batch_to_model_input(device_batch, device)
                user_ids = batch.get("user_id", [])

                logits = model.predict(model_input)
                probs = torch.sigmoid(logits.squeeze(-1)).float().cpu().tolist()
                predictions.update(
                    {_prediction_key(user_id): prob for user_id, prob in zip(user_ids, probs)}
                )

                if (batch_idx + 1) % 100 == 0:
                    logging.info(f"Processed {(batch_idx + 1) * batch_size} samples")
        return predictions

    logging.info("Starting inference...")
    try:
        predictions = collect_predictions(_make_data_loader(test_dataset, num_workers))
    except PermissionError:
        if num_workers <= 0:
            raise
        logging.warning(
            "DataLoader workers could not be started; retrying inference with "
            "num_workers=0."
        )
        predictions = collect_predictions(_make_data_loader(test_dataset, 0))

    output_path = os.path.join(result_dir, "predictions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"predictions": predictions}, f)

    logging.info(f"Saved {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
