#!/usr/bin/env python3
"""
Prints every parameter name in a model built from a config (改造計劃書.md §4.3).

encoder_prefixes/head_prefixes in a fine-tuning config must be *verified*
state_dict key prefixes, not guessed ones. MONAI's UNet is a nested
Sequential (parameter names look like model.model.model.0...), so a plausible
-looking guess like "model.encoder." will match nothing, and the training
run will silently fall back to a single LR group / load the entire
pretrained state_dict into the "head" with no warning that anything went
wrong. Run this once per model_type before writing encoder_prefixes/
head_prefixes/encoder_keep_prefixes into a fine-tuning config.

Usage:
    python scripts/inspect_model_params.py --config configs/config_cell.json
    python scripts/inspect_model_params.py --config configs/config_cell.json --model_type swin_unetr
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import build_model_from_config


def main():
    parser = argparse.ArgumentParser(description="Print named_parameters() for a model built from config, to find real encoder/head prefixes.")
    parser.add_argument("--config", type=str, required=True, help="Path to a train config (e.g. configs/config_cell.json)")
    parser.add_argument("--model_type", type=str, default=None, help="Override train.model_type from the config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        full_config = json.load(f)

    if args.model_type:
        full_config.setdefault("train", {})["model_type"] = args.model_type

    model_type = full_config.get("train", {}).get("model_type", "unet")
    model = build_model_from_config(full_config)

    names = [name for name, _ in model.named_parameters()]
    print(f"model_type={model_type}: {len(names)} parameters\n")
    for name in names:
        print(name)

    print(
        "\nPick prefixes from the names above for train.encoder_prefixes / train.head_prefixes "
        "in the fine-tuning config. A prefix that matches 0 names will silently do nothing at "
        "training time -- load_checkpoint_for_transfer()/train.py both warn if that happens, but "
        "verifying it here first is cheaper than a wasted training run."
    )


if __name__ == "__main__":
    main()
