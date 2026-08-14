#!/usr/bin/env python3
"""
Converts a legacy whole-object checkpoint (torch.save(model, path)) into the
format_version=2 state_dict + metadata format (改造計劃書.md §3.1/§4.2).

This lets existing trained weights (e.g. an already-trained c-Fos/TH model)
be used as a fine-tuning source under the new checkpoint format -- as a
candidate `shared_init` (計劃書 §5.1 source B), or just to get lineage/
preprocess-contract tracking on a marker's existing weights going forward.

Since the legacy checkpoint only contains a pickled nn.Module (no explicit
model_type/model_config), you supply the training config that was used to
build it, mirroring how train.py itself derives model_type/model_config.
The script sanity-checks that config against the legacy state_dict's actual
keys and warns (does not fail) on mismatch -- see the printed warning for
why a mismatch is expected specifically for architectures where
models/factory.py's enforce_instance_norm() changed the normalization layer
type (e.g. VNet: BatchNorm's running_mean/running_var/num_batches_tracked
buffers have no InstanceNorm equivalent). A legacy BatchNorm-trained model's
*weights* don't retroactively become InstanceNorm-trained weights just by
relabeling the layer -- this conversion preserves what was actually learned,
it doesn't fix the underlying BatchNorm-transfer risk for that specific old
checkpoint. New checkpoints trained from here on don't have this problem.

Usage:
    python scripts/convert_legacy_checkpoint.py \\
        --input output/TH/v12_unet_baseline/weights/v12_unet_baseline.pth \\
        --output output/TH/v12_unet_baseline/weights/v12_unet_baseline_v2.pth \\
        --config configs/config_cell.json \\
        --role marker_specific --marker TH
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models import build_model_from_config
from utils.checkpoint import save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("convert_legacy_checkpoint")


def main():
    parser = argparse.ArgumentParser(description="Convert a legacy torch.save(model, path) checkpoint to format_version=2.")
    parser.add_argument("--input", type=str, required=True, help="Path to the legacy .pth (whole-object) checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to write the format_version=2 .pth checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Train config that was used to build the legacy model (for model_type/model_config/preprocess)")
    parser.add_argument("--role", type=str, default="marker_specific", choices=["shared_init", "marker_specific"])
    parser.add_argument("--marker", type=str, default=None)
    parser.add_argument("--task_family", type=str, default=None)
    parser.add_argument("--parent", type=str, default=None, help="Path/id of the shared_init checkpoint this was originally fine-tuned from, if known")
    parser.add_argument("--shared_init_version", type=str, default=None)
    parser.add_argument("--n_annotated_crops", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_config = json.load(f)
    train_config = full_config.get("train", {})
    model_type = train_config.get("model_type", "unet")
    model_config = full_config.get("model", {}).get(model_type, {})
    preprocess = train_config.get("preprocess", {})

    legacy = torch.load(args.input, map_location="cpu", weights_only=False)
    if isinstance(legacy, dict) and "state_dict" in legacy:
        logger.error(f"{args.input} already looks like a format_version=2 checkpoint (has a 'state_dict' key) -- nothing to convert.")
        return 1
    if not hasattr(legacy, "state_dict"):
        logger.error(f"{args.input} did not load as an nn.Module (no .state_dict() method) -- unrecognized checkpoint format.")
        return 1

    # Sanity-check the supplied config against the legacy model's actual keys.
    reference_model = build_model_from_config(full_config)
    legacy_keys = set(legacy.state_dict().keys())
    reference_keys = set(reference_model.state_dict().keys())
    if legacy_keys != reference_keys:
        only_legacy = sorted(legacy_keys - reference_keys)
        only_reference = sorted(reference_keys - legacy_keys)
        logger.warning(
            f"state_dict keys differ between the legacy checkpoint and a model freshly built from "
            f"--config (model_type={model_type}). This can be expected (e.g. VNet's BatchNorm "
            f"running-stat buffers have no InstanceNorm equivalent after models/factory.py's "
            f"enforce_instance_norm), but double-check --config actually matches this checkpoint's "
            f"architecture.\n"
            f"  only in legacy checkpoint ({len(only_legacy)}): {only_legacy[:10]}{'...' if len(only_legacy) > 10 else ''}\n"
            f"  only in --config model ({len(only_reference)}): {only_reference[:10]}{'...' if len(only_reference) > 10 else ''}"
        )

    save_checkpoint(
        legacy, args.output,
        role=args.role,
        model_type=model_type,
        model_config=model_config,
        preprocess=preprocess,
        marker=args.marker,
        task_family=args.task_family,
        parent=args.parent,
        shared_init_version=args.shared_init_version,
        n_annotated_crops=args.n_annotated_crops,
        in_channels=model_config.get("in_channels", 1),
    )
    logger.info(f"Converted {args.input} -> {args.output} (role={args.role}, marker={args.marker}).")


if __name__ == "__main__":
    main()
