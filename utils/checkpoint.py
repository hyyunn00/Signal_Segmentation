"""Checkpoint v2: state_dict-based save/load with a lineage + preprocess
contract, replacing whole-object `torch.save(model, path)` (改造計劃書.md
§3.1/§4.2/§4.3).

Saving the whole nn.Module makes cross-marker transfer impossible: you can't
load only the encoder and drop the head, you can't freeze specific layers,
and the checkpoint breaks the moment models/ is refactored (pickle records
the class's full import path). This module stores state_dict + metadata
instead, so:
  - the same encoder weights can be partially loaded into a differently-headed
    model (fine-tuning),
  - the hub-and-spoke lineage rule (every marker checkpoint must trace back
    directly to a role="shared_init" checkpoint, not to another marker) can
    be enforced at load time instead of trusted by convention,
  - the preprocessing pipeline the encoder was trained with travels with the
    weights, so a silent normalize_mode mismatch between training runs fails
    loudly instead of quietly degrading results.
"""
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _get_git_commit() -> Optional[str]:
    """Best-effort commit hash for traceability. None if not in a git repo
    or git isn't available -- this must never fail a training run."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _get_env_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {"torch": getattr(torch, "__version__", None)}
    try:
        import monai
        versions["monai"] = getattr(monai, "__version__", None)
    except ImportError:
        versions["monai"] = None
    return versions


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    *,
    role: str,
    model_type: str,
    model_config: Dict[str, Any],
    preprocess: Dict[str, Any],
    marker: Optional[str] = None,
    task_family: Optional[str] = None,
    parent: Optional[str] = None,
    shared_init_version: Optional[str] = None,
    contributing_markers: Optional[List[str]] = None,
    n_annotated_crops: Optional[int] = None,
    seed: Optional[int] = None,
    in_channels: int = 1,
) -> None:
    """Saves model.state_dict() plus lineage/preprocess/provenance metadata.

    role must be "shared_init" (a hub checkpoint other markers fine-tune
    from) or "marker_specific" (a single marker's fine-tuned weights). See
    改造計劃書.md §2.1 for why chained marker-to-marker transfer is disallowed
    by convention -- load_checkpoint_for_transfer() enforces it at load time.
    """
    if role not in ("shared_init", "marker_specific"):
        raise ValueError(f"role must be 'shared_init' or 'marker_specific', got {role!r}")
    if role == "marker_specific" and parent is None:
        logger.warning(
            "Saving a marker_specific checkpoint with parent=None -- this weight's lineage "
            "back to a shared_init checkpoint will be unrecoverable later. Pass parent= if "
            "this run fine-tuned from a shared_init checkpoint."
        )

    checkpoint = {
        "format_version": 2,
        "state_dict": model.state_dict(),

        "role": role,
        "parent": parent,
        "shared_init_version": shared_init_version,

        "model_type": model_type,
        "model_config": model_config,
        "in_channels": in_channels,

        "marker": marker,
        "task_family": task_family,
        "contributing_markers": contributing_markers,
        "n_annotated_crops": n_annotated_crops,

        "preprocess": preprocess,

        "seed": seed,
        "commit": _get_git_commit(),
        "env": _get_env_versions(),
    }
    torch.save(checkpoint, path)
    logger.info(f"[OK] Checkpoint (format_version=2, role={role}) saved to {path}")


def load_checkpoint_for_transfer(path: str, model: torch.nn.Module, config: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Loads a checkpoint into `model` for fine-tuning, enforcing the
    hub-and-spoke lineage rule and the preprocess contract, then drops the
    output head (and optionally the deepest encoder blocks) so they get
    freshly initialized instead of overwritten with a mismatched shape/task.

    config (the "train" section) is expected to provide:
      - preprocess: dict, must match the checkpoint's preprocess exactly
      - allow_chained_transfer: bool, required to load a marker_specific
        checkpoint instead of a shared_init one (default False)
      - head_prefixes: list[str] of state_dict key prefixes to drop so the
        output head is reinitialized (see scripts/inspect_model_params.py)
      - reset_head: bool, default True
      - encoder_prefixes / encoder_keep_prefixes: optional, drops encoder
        keys under encoder_prefixes that don't also start with one of
        encoder_keep_prefixes -- i.e. "only load the first N encoder
        blocks, let the rest re-initialize" (改造計劃書.md §2.3's truncated
        encoder). Both must be explicit key prefixes verified against the
        actual model (via scripts/inspect_model_params.py), not guessed --
        MONAI's UNet is a nested Sequential, so a wrong guess fails silently.

    Returns (missing_keys, unexpected_keys, checkpoint_metadata) -- the last
    is the loaded checkpoint dict minus its (large) state_dict, so callers
    that want to propagate lineage into their own saved checkpoint (e.g.
    parent=path, shared_init_version=checkpoint_metadata["shared_init_version"])
    don't need to re-read the file from disk.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "state_dict" not in ck:
        raise ValueError(
            f"{path} is not a format_version=2 checkpoint (expected a dict with a 'state_dict' "
            f"key). Legacy whole-object checkpoints must be converted first with "
            f"scripts/convert_legacy_checkpoint.py."
        )

    role = ck.get("role")
    if role != "shared_init" and not config.get("allow_chained_transfer", False):
        raise ValueError(
            f"pretrained_weights role='{role}' (marker={ck.get('marker')}). This project uses "
            f"hub-and-spoke transfer: fine-tuning should start from a role='shared_init' "
            f"checkpoint, not from another marker's weights, to avoid chained-bias accumulation "
            f"(改造計劃書.md §2.1). If chained transfer is really intended, set "
            f"allow_chained_transfer=true in the train config."
        )

    ckpt_preprocess = ck.get("preprocess")
    cfg_preprocess = config.get("preprocess")
    if ckpt_preprocess != cfg_preprocess:
        raise ValueError(
            f"preprocess contract mismatch between checkpoint and config -- refusing to "
            f"fine-tune with a silently different normalization pipeline than the one the "
            f"encoder was trained with (改造計劃書.md §4.2).\n"
            f"  checkpoint preprocess: {ckpt_preprocess}\n"
            f"  config preprocess:     {cfg_preprocess}"
        )

    sd = dict(ck["state_dict"])

    head_prefixes = config.get("head_prefixes", [])
    if config.get("reset_head", True) and head_prefixes:
        sd = {k: v for k, v in sd.items() if not any(k.startswith(p) for p in head_prefixes)}

    encoder_prefixes = config.get("encoder_prefixes", [])
    encoder_keep_prefixes = config.get("encoder_keep_prefixes")
    if encoder_keep_prefixes and encoder_prefixes:
        def _is_dropped_encoder_key(k: str) -> bool:
            is_encoder = any(k.startswith(p) for p in encoder_prefixes)
            return is_encoder and not any(k.startswith(p) for p in encoder_keep_prefixes)
        sd = {k: v for k, v in sd.items() if not _is_dropped_encoder_key(k)}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(
        f"[transfer] from={path} shared_init_version={ck.get('shared_init_version')} "
        f"loaded_keys={len(sd)} missing={len(missing)} unexpected={len(unexpected)}"
    )
    if len(sd) == 0:
        logger.warning(
            "[transfer] loaded 0 keys from checkpoint -- head_prefixes/encoder_prefixes are "
            "almost certainly misconfigured. Run scripts/inspect_model_params.py against this "
            "model_type first to get the real parameter name prefixes."
        )

    # Metadata minus the (large) state_dict, so callers that want to record
    # lineage in their own saved checkpoint (e.g. train.py setting
    # parent=pretrained_path, shared_init_version=ck["shared_init_version"])
    # don't need to re-read the checkpoint from disk a second time.
    meta = {k: v for k, v in ck.items() if k != "state_dict"}
    return missing, unexpected, meta


def load_checkpoint_for_inference(path: str, map_location: str = "cpu") -> torch.nn.Module:
    """Rebuilds a model from its stored model_type/model_config and loads
    weights strictly. Only handles format_version=2 checkpoints -- callers
    should fall back to the legacy torch.load(..., weights_only=False) path
    for old whole-object checkpoints (see inference.py::load_checkpoint)."""
    from models import build_model_from_config

    ck = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ck, dict) or "state_dict" not in ck:
        raise ValueError(f"{path} is not a format_version=2 checkpoint.")

    sub_config = dict(ck["model_config"])
    sub_config["model_type"] = ck["model_type"]
    sub_config.setdefault("in_channels", ck.get("in_channels", 1))

    model = build_model_from_config(sub_config)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()
    logger.info(f"[OK] Loaded format_version=2 checkpoint from {path} (model_type={ck['model_type']}, role={ck.get('role')}, marker={ck.get('marker')})")
    return model
