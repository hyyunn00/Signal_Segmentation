"""Global reproducibility control (改造計劃書.md §3.5).

Without this, two runs of the same config produce different results because
no RNG is seeded anywhere (train.py, DataLoader workers, utils/cropper.py's
global np.random calls). That makes the annotation-efficiency experiment in
§7 (24 training arms compared against each other) impossible to read, since
run-to-run noise can be as large as the effect being measured.
"""
import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seeds python/numpy/torch RNGs.

    `deterministic` controls the cudnn.deterministic/benchmark tradeoff:
    - False (default): cudnn.benchmark=True, so cudnn auto-tunes convolution
      algorithms for throughput. This is what regular production training
      should use on the A6000 -- reproducibility isn't free, and most runs
      don't need bit-exact repeatability.
    - True: cudnn.deterministic=True, cudnn.benchmark=False. Use this for the
      controlled comparisons in §7 (same config, different seeds, compared
      against each other) where run-to-run repeatability matters more than
      raw throughput.

    Must be called before dataset construction: utils/cropper.py's
    filter_indices_by_mask() uses the global np.random state (not an
    explicitly-passed generator), and that call happens single-threaded in
    the main process before any DataLoader worker is forked, so seeding the
    global state here is sufficient to make it deterministic too.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    logger.info(f"Global seed set to {seed} (deterministic={deterministic}).")


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: reseeds numpy/random per-worker from torch's
    per-worker base seed so worker processes don't all share the parent's
    RNG state (torch's own RNG is already reseeded per-worker by DataLoader).
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
