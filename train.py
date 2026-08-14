#!/usr/bin/env python3
"""
Train a 2D/3D U-Net-style segmentation model on microscopy data using shared memory.
"""
import warnings
# Suppress the cuda.cudart module deprecation warning (must be done before other imports)
warnings.filterwarnings("ignore", category=FutureWarning, module="cuda")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import argparse
import os
import torch
import torch.optim as optim
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Callable, Union

from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.spatial.dictionary import RandFlipd
from monai.transforms.intensity.dictionary import (
    GaussianSmoothd, NormalizeIntensityd, RandAdjustContrastd, RandBiasFieldd, 
    RandShiftIntensityd, RandScaleIntensityd, RandGaussianNoised
)
from monai.transforms.post.dictionary import AsDiscreted
from monai.data.dataloader import DataLoader

from IO import build_train_dataset_from_config
from models import build_model_from_config
from utils.visualization import visualize_dataset, visualize_predictions
from utils.metrics import build_metrics_from_config
from utils.plot import save_learning_curves
from utils.concurrency import initialize_concurrency
from utils.loss import build_loss_from_config
from utils.seeding import set_global_seed, seed_worker
from utils.checkpoint import save_checkpoint as save_checkpoint_v2, load_checkpoint_for_transfer

# Initialize logging
logger = logging.getLogger(__name__)

# Transforms
train_transform = Compose([
    ToTensord(keys=["image", "mask"], dtype=torch.float32),
    # GaussianSmoothd(keys=["mask"], sigma=0.1),
    AsDiscreted(keys=["mask"], threshold=0.5),
    RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
    RandAdjustContrastd(keys=["image"], prob=0.3),
    # RandGaussianNoised(keys=["image"], prob=0.4, mean=0.0, std=0.1),
    RandBiasFieldd(keys=["image"], prob=0.2),
    RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.3),
    RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.3),
])

val_transform = Compose([
    ToTensord(keys=["image", "mask"], dtype=torch.float32),
    # GaussianSmoothd(keys=["mask"], sigma=0.1),
    AsDiscreted(keys=["mask"], threshold=0.5),
])

def save_checkpoint(
    model: torch.nn.Module,
    weight_path: str,
    name: str,
    *,
    train_config: dict,
    model_type: str,
    model_config: dict,
    parent: Optional[str] = None,
    shared_init_version: Optional[str] = None,
):
    """Saves the model checkpoint in format_version=2 (state_dict + lineage +
    preprocess-contract metadata; see utils/checkpoint.py). Replaces the old
    whole-object torch.save(model, path), which made partial loading
    (encoder-only transfer) and layer freezing impossible."""
    path = os.path.join(weight_path, f"{name}.pth")
    save_checkpoint_v2(
        model, path,
        role=train_config.get("role", "marker_specific"),
        model_type=model_type,
        model_config=model_config,
        preprocess=train_config.get("preprocess", {}),
        marker=train_config.get("marker"),
        task_family=train_config.get("task_family"),
        parent=parent,
        shared_init_version=shared_init_version,
        contributing_markers=train_config.get("contributing_markers"),
        n_annotated_crops=train_config.get("n_annotated_crops"),
        seed=train_config.get("seed"),
        in_channels=model_config.get("in_channels", 1),
    )

def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    metrics: Dict[str, torch.nn.Module],
    device: torch.device,
    epoch: int,
    max_grad_norm: Optional[float] = None
) -> Dict[str, float]:
    """Runs a single training epoch."""
    model.train()
    desc = f"Training Epoch {epoch+1}"
    
    metric_sums: Dict[str, float] = {m: 0.0 for m in metrics.keys()}
    total_loss = 0.0
    n_batches = max(1, len(loader))

    progress = tqdm(
        loader, 
        desc=desc, 
        leave=False, 
        bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss_val = criterion(outputs, masks)
        loss_val.backward()
        
        # Gradient Clipping
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
        optimizer.step()

        batch_loss = float(loss_val.item())
        total_loss += batch_loss
        
        current_metrics = {"loss": batch_loss}
        for name, fn in metrics.items():
            try:
                v = fn(outputs, masks)
                val = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                current_metrics[name] = val
                metric_sums[name] += val
            except Exception as e:
                logger.warning(f"Metric '{name}' failed on this batch: {e}")

        progress.set_postfix({k: f"{v:.4f}" for k, v in current_metrics.items()})

    results = {"loss": total_loss / n_batches}
    for m in metrics.keys():
        results[m] = metric_sums[m] / n_batches
    return results

def valid_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    metrics: Dict[str, torch.nn.Module],
    device: torch.device,
    epoch: int,
    viz_path: Optional[str] = None,
    cache_size: int = 20
) -> Dict[str, float]:
    """Runs a single validation epoch and optionally handles visualization."""
    model.eval()
    desc = f"Validating Epoch {epoch+1}"
    
    metric_sums: Dict[str, float] = {m: 0.0 for m in metrics.keys()}
    total_loss = 0.0
    n_batches = max(1, len(loader))

    # Visualization caching every 25 epochs
    is_viz_epoch = (epoch + 1) % 25 == 0 and viz_path is not None
    viz_cache = {"images": [], "masks": [], "outputs": []} if is_viz_epoch else None

    with torch.no_grad():
        progress = tqdm(
            loader, 
            desc=desc, 
            leave=False, 
            bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            loss_val = criterion(outputs, masks)

            batch_loss = float(loss_val.item())
            total_loss += batch_loss
            
            current_metrics = {"loss": batch_loss}
            for name, fn in metrics.items():
                try:
                    v = fn(outputs, masks)
                    val = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                    current_metrics[name] = val
                    metric_sums[name] += val
                except Exception as e:
                    logger.warning(f"Metric '{name}' failed on this batch: {e}")

            if is_viz_epoch and viz_cache:
                current_count = len(viz_cache["images"])
                if current_count < cache_size:
                    n_to_add = min(cache_size - current_count, images.size(0))
                    viz_cache["images"].extend(images[:n_to_add].detach().cpu().numpy())
                    viz_cache["masks"].extend(masks[:n_to_add].detach().cpu().numpy())
                    viz_cache["outputs"].extend(outputs[:n_to_add].detach().cpu().numpy())

            progress.set_postfix({k: f"{v:.4f}" for k, v in current_metrics.items()})

    if is_viz_epoch and viz_cache and len(viz_cache["images"]) > 0:
        visualize_predictions(
            np.array(viz_cache["images"]),
            np.array(viz_cache["masks"]),
            np.array(viz_cache["outputs"]),
            save_path=viz_path,
            title=f"Epoch_{epoch+1}_Validation"
        )

    results = {"loss": total_loss / n_batches}
    for m in metrics.keys():
        results[m] = metric_sums[m] / n_batches
    return results

def main():
    parser = argparse.ArgumentParser(description="Train U-Net for Microscopy Segmentation")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        full_config = json.load(f)
    
    # Initialize concurrency settings
    initialize_concurrency(full_config)
        
    config = full_config.get("train", {})
    model_config = full_config.get("model", {})

    # Seed every RNG (python/numpy/torch) before any dataset construction or
    # model init happens, so a config produces the same result run-to-run.
    # See utils/seeding.py for why this must happen before
    # build_train_dataset_from_config() specifically.
    seed = config.get("seed", 42)
    set_global_seed(seed, deterministic=config.get("deterministic", False))

    img_root, mask_root = config.get("img_path"), config.get("mask_path")
    data_root = config.get("data_path")
    save_root = config.get("save_path")
    model_name = config.get("model_name", "best_model")
    
    if not (data_root or (img_root and mask_root)) or not save_root:
        logging.error("Missing mandatory paths (data_path or img/mask_path) in config."); return 1
        
    # Paths setup
    model_save_path = os.path.join(save_root, model_name)
    viz_path = os.path.join(model_save_path, "visualization")
    weight_path = os.path.join(model_save_path, "weights")
    artifact_path = os.path.join(model_save_path, "artifacts")
    for p in [viz_path, weight_path, artifact_path]: os.makedirs(p, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(artifact_path, f"train_{timestamp}.log")
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Archive config and model source
    import shutil
    shutil.copy2(args.config, os.path.join(artifact_path, "config.json"))
    
    model_type = config.get("model_type", "unet")
    model_source_map = {
        "unet": "UNet.py",
        "attention_unet": "AttentionUNet.py",
        "swin_unetr": "SwinUNETR.py",
        "vnet": "VNet.py"
    }
    
    if model_type in model_source_map:
        model_src = os.path.join("models", model_source_map[model_type])
        if os.path.exists(model_src):
            shutil.copy2(model_src, os.path.join(artifact_path, model_source_map[model_type]))

    # Dataset & Dataloaders
    train_ds, val_ds = build_train_dataset_from_config(full_config, train_transform, val_transform)
    
    if config.get("visualize_preview", False):
        visualize_dataset(train_ds, title="train_samples_preview", save_path=viz_path)
        visualize_dataset(val_ds, title="validation_samples_preview", save_path=viz_path)
    
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.get("training_batch_size", 8),
        shuffle=True,
        num_workers=config.get("training_num_workers", 4),
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=loader_generator
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.get("training_batch_size", 8),
        shuffle=False,
        num_workers=config.get("training_num_workers", 4),
        persistent_workers=True,
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    
    # Model
    patch_size = config.get("training_patch_size", [16, 64, 64])
    spatial_dims = 3 if patch_size[0] > 1 else 2
    
    if model_type in full_config.get("model", {}):
        full_config["model"][model_type]["spatial_dims"] = spatial_dims

    model = build_model_from_config(full_config)
    model_config_used = full_config.get("model", {}).get(model_type, {})
    criterion = build_loss_from_config(full_config)
    metrics = build_metrics_from_config(full_config)

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    # ---- Fine-tuning load path (改造計劃書.md §3.2/§4.2/§4.3) ----
    # Loads a format_version=2 shared_init/marker_specific checkpoint,
    # enforcing the hub-and-spoke lineage rule and the preprocess contract,
    # then drops the output head so it re-initializes for this marker/task.
    pretrained_path = config.get("pretrained_weights")
    parent_for_checkpoint = None
    shared_init_version_used = None
    if pretrained_path:
        missing, unexpected, ckpt_meta = load_checkpoint_for_transfer(pretrained_path, model, config)
        model.to(device)  # state_dict was loaded on CPU; re-affirm device placement
        parent_for_checkpoint = pretrained_path
        shared_init_version_used = ckpt_meta.get("shared_init_version")
        logging.info(
            f"[transfer] Fine-tuning from {pretrained_path} "
            f"(shared_init_version={shared_init_version_used}, missing={len(missing)}, unexpected={len(unexpected)})."
        )

    # ---- Learning-rate grouping (改造計劃書.md §4.4) ----
    # encoder_prefixes must be explicit state_dict key prefixes the user has
    # verified against this model_type (see scripts/inspect_model_params.py)
    # -- MONAI's UNet is a nested Sequential, so a guessed prefix would
    # silently match nothing and this would quietly fall back to a single LR
    # group. When encoder_prefixes isn't set (the default for existing
    # from-scratch configs), all parameters go in one group and behavior is
    # unchanged from before.
    transfer_cfg = config.get("transfer", {})
    encoder_prefixes = config.get("encoder_prefixes", [])
    encoder_params, other_params = [], []
    if encoder_prefixes:
        for pname, p in model.named_parameters():
            (encoder_params if any(pname.startswith(pfx) for pfx in encoder_prefixes) else other_params).append(p)
        if not encoder_params:
            logging.warning(
                "encoder_prefixes did not match any parameter names -- falling back to a single "
                "LR group for all parameters. Run scripts/inspect_model_params.py to get the real "
                "prefixes for this model_type before relying on encoder_lr_scale/staged unfreezing."
            )
            other_params = list(model.parameters())
    else:
        other_params = list(model.parameters())

    base_lr = config.get("learning_rate", 1e-4)
    default_encoder_scale = 0.1 if pretrained_path else 1.0
    encoder_lr_scale = transfer_cfg.get("encoder_lr_scale", default_encoder_scale)

    if encoder_params:
        optimizer = optim.AdamW([
            {"params": encoder_params, "lr": base_lr * encoder_lr_scale, "name": "encoder"},
            {"params": other_params, "lr": base_lr, "name": "other"},
        ], weight_decay=config.get("weight_decay", 1e-5))
    else:
        optimizer = optim.AdamW(other_params, lr=base_lr, weight_decay=config.get("weight_decay", 1e-5))

    # ---- Staged unfreeze schedule (改造計劃書.md §4.3) ----
    # Only engages when fine-tuning from a checkpoint AND encoder params were
    # actually identified -- from-scratch training keeps the original
    # ReduceLROnPlateau behavior untouched. Skipping phase 1 (frozen encoder)
    # is the most common way transfer learning "doesn't work": the freshly
    # random head produces large gradients on the first few batches that
    # wash out the pretrained encoder features before the head has aligned
    # to them at all.
    total_epochs = config.get("training_epochs", 30)
    use_staged_transfer = bool(pretrained_path) and bool(encoder_params)
    if use_staged_transfer:
        freeze_epochs = transfer_cfg.get("freeze_epochs", 20)
        finetune_epochs = transfer_cfg.get("finetune_epochs", 60)
        warmup_epochs = max(1, transfer_cfg.get("warmup_epochs", 5))
        phase_lrs = transfer_cfg.get("phase_lrs", [1e-3, 1e-4, 1e-5])
        scheduler = None  # LR is driven manually per-epoch below instead of ReduceLROnPlateau
        logging.info(
            f"[transfer] Staged unfreeze: phase1 (encoder frozen) epoch<{freeze_epochs}, "
            f"phase2 (encoder lr x{encoder_lr_scale}) epoch<{freeze_epochs + finetune_epochs}, "
            f"phase3 (full network, lower lr) after. warmup_epochs={warmup_epochs}, phase_lrs={phase_lrs}."
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history: Dict[str, Dict[str, List[float]]] = {n: {"train": [], "val": []} for n in list(metrics.keys()) + ["loss"]}
    best_val_loss = float("inf")

    # Training Loop
    logging.info("Starting training...")
    max_grad_norm = config.get("max_grad_norm")
    metric_interval = config.get("metric_interval", 1)
    
    # Separate metrics: only use torch-based nn.Module metrics for training if any, 
    # but based on user request, let's just use loss for training to be fastest.
    # We will compute all metrics only during validation.
    
    for epoch in range(config.get("training_epochs", 30)):
        print("\n"); logger.info(f"Epoch {epoch + 1}")

        if use_staged_transfer:
            if epoch < freeze_epochs:
                phase = 1
            elif epoch < freeze_epochs + finetune_epochs:
                phase = 2
            else:
                phase = 3
            phase_lr = phase_lrs[min(phase - 1, len(phase_lrs) - 1)]

            for p in encoder_params:
                p.requires_grad = (phase != 1)

            if epoch < warmup_epochs:
                lr_mult = (epoch + 1) / warmup_epochs
            else:
                cosine_progress = min(1.0, (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs))
                lr_mult = 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * cosine_progress))

            for group in optimizer.param_groups:
                scale = encoder_lr_scale if group.get("name") == "encoder" else 1.0
                group["lr"] = phase_lr * scale * lr_mult

            if epoch in (0, freeze_epochs, freeze_epochs + finetune_epochs):
                logger.info(f"[transfer] Epoch {epoch + 1}: entering phase {phase} (encoder frozen={phase == 1}, phase_lr={phase_lr}).")

        # Calculate heavy metrics only on interval epochs
        is_metric_epoch = (epoch + 1) % metric_interval == 0
        curr_metrics = metrics if is_metric_epoch else {}
        
        # Training
        train_results = train_epoch(model, train_loader, optimizer, criterion, curr_metrics, device, epoch, max_grad_norm)
        
        # Validation
        val_results = valid_epoch(model, val_loader, criterion, curr_metrics, device, epoch, viz_path=viz_path)
        
        for k in history.keys():
            if k == "loss":
                history[k]["train"].append(train_results["loss"])
                history[k]["val"].append(val_results["loss"])
            else:
                # For metrics, if not calculated this epoch, carry over the last known value
                # This keeps the learning curves continuous in plots
                t_val = train_results.get(k)
                if t_val is not None:
                    history[k]["train"].append(t_val)
                else:
                    prev = history[k]["train"][-1] if history[k]["train"] else 0.0
                    history[k]["train"].append(prev)
                
                v_val = val_results.get(k)
                if v_val is not None:
                    history[k]["val"].append(v_val)
                else:
                    prev = history[k]["val"][-1] if history[k]["val"] else 0.0
                    history[k]["val"].append(prev)
            
        logger.info(f"Loss -> Train: {train_results['loss']:.4f} | Val: {val_results['loss']:.4f}")
        for m in metrics.keys():
            log_parts = []
            if m in train_results:
                log_parts.append(f"Train: {train_results[m]:.4f}")
            if m in val_results:
                log_parts.append(f"Val: {val_results[m]:.4f}")
            
            if log_parts:
                logger.info(f"{m.capitalize()} -> {' | '.join(log_parts)}")

        val_avg_loss = val_results["loss"]

        if scheduler is not None:
            scheduler.step(val_avg_loss)
        checkpoint_kwargs = dict(
            train_config=config, model_type=model_type, model_config=model_config_used,
            parent=parent_for_checkpoint, shared_init_version=shared_init_version_used,
        )
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss; save_checkpoint(model, weight_path, model_name, **checkpoint_kwargs)

        if (epoch + 1) % 25 == 0:
            save_checkpoint(model, weight_path, f"{model_name}_epoch_{epoch+1}", **checkpoint_kwargs)

        save_learning_curves(history, artifact_path, model_name)

    logging.info("Training complete.")

if __name__ == "__main__":
    main()
