"""Config-driven train/val augmentation (改造計劃書.md §8).

Previously the transform pipeline was hardcoded as module-level constants at
the top of train.py -- the one part of this project not config-driven (loss,
metrics, model, and normalization all are, see utils/loss.py,
utils/metrics.py, models/factory.py, utils/normalization.py). That meant
augmentation strategy couldn't be swept, recorded into run artifacts, or
varied per marker without editing train.py's source directly.

Also fixes three gaps the plan calls out in the old hardcoded pipeline:
  - RandGaussianNoised was present but commented out (model never saw
    varying signal-to-noise ratio)
  - RandFlipd only covered one spatial axis (these volumes have no
    privileged orientation, so that discarded two free invariances)
  - no blur augmentation at all

Note: enabling stronger augmentation is expected to make the in-domain
validation loss look worse -- that's the augmentation doing its job (the
model now has to work slightly harder on the training distribution too).
Whether it actually helped must be judged from held-out detection metrics
(analysis3d.py), not from validation loss/Dice moving.
"""
import torch
from typing import Any, Dict, Optional

from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.spatial.dictionary import RandFlipd, RandAffined
from monai.transforms.intensity.dictionary import (
    RandAdjustContrastd, RandBiasFieldd, RandShiftIntensityd, RandScaleIntensityd,
    RandGaussianNoised, RandGaussianSmoothd,
)
from monai.transforms.post.dictionary import AsDiscreted

DEFAULT_AUGMENTATION_CONFIG: Dict[str, Any] = {
    "flip_prob": 0.5,

    "rotate_prob": 0.3,
    "rotate_range_rad": 0.15,   # ~8.5 degrees per axis; modest general-purpose default
    "shear_range": 0.0,         # set > 0 to add shear (useful for fiber/vessel markers)

    "adjust_contrast_prob": 0.3,
    "gaussian_noise_prob": 0.3,
    "gaussian_noise_std": 0.1,
    "bias_field_prob": 0.2,
    "shift_intensity_prob": 0.3,
    "shift_intensity_offset": 0.2,
    "scale_intensity_prob": 0.3,
    "scale_intensity_factor": 0.2,

    # Anisotropic blur / simulated coarse-z-resolution (改造計劃書.md §8 and
    # §9.5's "階梯一": the zero-cost version of PSF domain randomization).
    # Off by default (prob=0) so existing configs are unaffected -- opt in
    # per marker via train.augmentation.anisotropic_blur_prob.
    "anisotropic_blur_prob": 0.0,
    "anisotropic_blur_sigma_xy": [0.25, 1.0],
    "anisotropic_blur_sigma_z": [0.5, 2.0],
}


def build_train_transform(aug_config: Optional[Dict[str, Any]] = None) -> Compose:
    """Builds the training augmentation pipeline. aug_config overrides
    DEFAULT_AUGMENTATION_CONFIG (typically full_config["train"]["augmentation"])."""
    cfg = {**DEFAULT_AUGMENTATION_CONFIG, **(aug_config or {})}

    transforms = [
        ToTensord(keys=["image", "mask"], dtype=torch.float32),
        AsDiscreted(keys=["mask"], threshold=0.5),

        # Three-axis flip: the old pipeline only flipped spatial_axis=1.
        # These volumes have no privileged orientation, so restricting to one
        # axis wasted two free invariances the model should get for free.
        RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=cfg["flip_prob"]),
        RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=cfg["flip_prob"]),
        RandFlipd(keys=["image", "mask"], spatial_axis=2, prob=cfg["flip_prob"]),

        RandAffined(
            keys=["image", "mask"],
            prob=cfg["rotate_prob"],
            rotate_range=(cfg["rotate_range_rad"],) * 3,
            shear_range=(cfg["shear_range"],) * 3,
            mode=("bilinear", "nearest"),  # image: smooth interpolation; mask: must stay binary
            padding_mode="zeros",
        ),

        RandAdjustContrastd(keys=["image"], prob=cfg["adjust_contrast_prob"]),
        RandGaussianNoised(keys=["image"], prob=cfg["gaussian_noise_prob"], mean=0.0, std=cfg["gaussian_noise_std"]),
        RandBiasFieldd(keys=["image"], prob=cfg["bias_field_prob"]),
        RandShiftIntensityd(keys=["image"], offsets=cfg["shift_intensity_offset"], prob=cfg["shift_intensity_prob"]),
        RandScaleIntensityd(keys=["image"], factors=cfg["scale_intensity_factor"], prob=cfg["scale_intensity_prob"]),
    ]

    if cfg["anisotropic_blur_prob"] > 0:
        transforms.append(RandGaussianSmoothd(
            keys=["image"],
            sigma_x=tuple(cfg["anisotropic_blur_sigma_xy"]),
            sigma_y=tuple(cfg["anisotropic_blur_sigma_xy"]),
            sigma_z=tuple(cfg["anisotropic_blur_sigma_z"]),
            prob=cfg["anisotropic_blur_prob"],
        ))

    return Compose(transforms)


def build_val_transform() -> Compose:
    """No augmentation for validation -- just tensor conversion + mask binarization."""
    return Compose([
        ToTensord(keys=["image", "mask"], dtype=torch.float32),
        AsDiscreted(keys=["mask"], threshold=0.5),
    ])
