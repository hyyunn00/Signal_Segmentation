"""Enforces InstanceNorm across model architectures (改造計劃書.md §4.5).

BatchNorm bakes the *source* dataset's batch statistics (running mean/var)
into the weights. Loaded into a new marker, those statistics describe
another signal's brightness distribution and quietly poison the features --
this is the classic, silent way cross-domain transfer fails: there's no
error, training just doesn't transfer. InstanceNorm normalizes per-sample
and carries no dataset-level statistics, so it's the right default for
weights meant to be reused across markers.

Rather than trust each MONAI architecture's current default norm layer
(which differs by class and can change across MONAI versions -- see
改造計劃書.md §3.6's dependency-risk note on SwinUNETR), this walks the built
model and replaces any BatchNorm found with an equivalent InstanceNorm, so
the guarantee holds regardless of upstream defaults. VNet is the one model
in this project that hardcodes BatchNorm3d internally with no constructor
option to switch it, making this a real fix and not just a defensive no-op.
"""
import logging

import torch.nn as nn

logger = logging.getLogger(__name__)

_BATCHNORM_TO_INSTANCENORM = {
    nn.BatchNorm1d: nn.InstanceNorm1d,
    nn.BatchNorm2d: nn.InstanceNorm2d,
    nn.BatchNorm3d: nn.InstanceNorm3d,
}


def enforce_instance_norm(model: nn.Module) -> nn.Module:
    """Replaces every nn.BatchNorm{1,2,3}d submodule in-place with a freshly
    initialized nn.InstanceNorm{1,2,3}d(affine=True) of the same channel
    count. Returns the same model object (mutated in place) for chaining.
    """
    replaced = 0
    for parent_name, parent_module in model.named_modules():
        for child_name, child in list(parent_module.named_children()):
            instance_cls = _BATCHNORM_TO_INSTANCENORM.get(type(child))
            if instance_cls is None:
                continue
            replacement = instance_cls(child.num_features, affine=True)
            setattr(parent_module, child_name, replacement)
            replaced += 1
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            logger.debug(f"[norm] Replaced BatchNorm -> InstanceNorm at {full_name}")

    if replaced:
        logger.info(f"[norm] Converted {replaced} BatchNorm layer(s) to InstanceNorm.")
    return model
