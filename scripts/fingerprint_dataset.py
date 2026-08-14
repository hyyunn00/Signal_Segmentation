#!/usr/bin/env python3
"""
Dataset fingerprinting (改造計劃書.md §6.3): measures object size distribution,
foreground voxel fraction, object count, and a shape (elongation) indicator
directly from mask volumes, instead of having someone guess these numbers by
eye per marker.

These measurements are meant to drive config decisions automatically rather
than by hand:
  - equivalent diameter (voxels)  -> whether to truncate the encoder depth
                                      (改造計劃書.md §2.3) and detection-match
                                      tolerance (§11, "容差取物件半徑")
  - foreground voxel fraction     -> Tversky alpha/beta, neg_keep_ratio,
                                      focal loss weighting
  - elongation                    -> task_family (blob vs tubular), which in
                                      turn selects the output head/loss/metrics

Shape/size stats are computed from each connected component's raw voxel
coordinates (region.coords) and voxel count (region.area) only -- not from
skimage's ND-only regionprops fields (equivalent_diameter, inertia_tensor_*,
etc.), whose availability/naming has changed across skimage versions
(改造計劃書.md §3.6's dependency-risk note). Elongation is instead computed
directly from the eigenvalues of each object's voxel-coordinate covariance
matrix via numpy, which has no such version dependency.

Usage:
    python scripts/fingerprint_dataset.py --data_root /path/to/training-data
    python scripts/fingerprint_dataset.py --data_root ... --mask_name Flatten_561_mask --spacing_um 2.0 1.8 1.8
"""
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from skimage.measure import label, regionprops
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from IO.reader import FileReader
from analysis3d import _filter_small_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("fingerprint_dataset")

DETAILED_HEADERS = [
    "parent_folder", "volume_name", "n_objects", "foreground_voxel_fraction",
    "diameter_voxel_mean", "diameter_voxel_median", "diameter_voxel_p5", "diameter_voxel_p95",
    "diameter_um_mean", "diameter_um_median",
    "elongation_mean", "elongation_median",
    "task_family_guess",
]

ELONGATION_TUBULAR_THRESHOLD = 2.5  # median elongation above this -> tubular-leaning shape mix


def _object_elongation(coords: np.ndarray) -> Optional[float]:
    """Ratio of largest to smallest principal spread of an object's voxel
    coordinates -- close to 1 for a blob, larger for an elongated/tubular
    fragment. None if there aren't enough voxels for a stable estimate."""
    if coords.shape[0] < 4:
        return None
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 1e-9, None)
    return float(np.sqrt(eigvals[-1] / eigvals[0]))


def fingerprint_volume(mask: np.ndarray, spacing_um: Optional[List[float]] = None, min_size: int = 0) -> Optional[dict]:
    """Measures object size distribution, foreground fraction, and shape for
    a single mask volume. Returns None if the mask has no objects."""
    labels = label(mask > 0.5, connectivity=3)
    if min_size > 0:
        labels = _filter_small_labels(labels, min_size)

    n_objects = int(labels.max())
    if n_objects == 0:
        return None

    regions = regionprops(labels)
    voxel_counts = np.array([r.area for r in regions], dtype=np.float64)
    diameters_voxel = (6.0 * voxel_counts / np.pi) ** (1.0 / 3.0)

    elongations = [e for e in (_object_elongation(r.coords) for r in regions) if e is not None]

    result = {
        "n_objects": n_objects,
        "foreground_voxel_fraction": float(np.mean(mask > 0.5)),
        "diameter_voxel_mean": float(np.mean(diameters_voxel)),
        "diameter_voxel_median": float(np.median(diameters_voxel)),
        "diameter_voxel_p5": float(np.percentile(diameters_voxel, 5)),
        "diameter_voxel_p95": float(np.percentile(diameters_voxel, 95)),
        "elongation_mean": float(np.mean(elongations)) if elongations else None,
        "elongation_median": float(np.median(elongations)) if elongations else None,
    }

    if spacing_um is not None:
        # Approximate physical diameter using the mean voxel spacing --
        # objects aren't generally axis-aligned, so this is a scale
        # estimate, not an exact physical measurement.
        mean_spacing = float(np.mean(spacing_um))
        result["diameter_um_mean"] = result["diameter_voxel_mean"] * mean_spacing
        result["diameter_um_median"] = result["diameter_voxel_median"] * mean_spacing
    else:
        result["diameter_um_mean"] = None
        result["diameter_um_median"] = None

    median_elong = result["elongation_median"]
    if median_elong is None:
        result["task_family_guess"] = "unknown"
    elif median_elong >= ELONGATION_TUBULAR_THRESHOLD:
        result["task_family_guess"] = "tubular"
    else:
        result["task_family_guess"] = "blob"

    return result


def discover_mask_volumes(root: Path, mask_name: str) -> List[Path]:
    return sorted(p for p in root.rglob(mask_name) if p.is_dir())


def write_report(rows: List[dict], out_path: Path):
    try:
        import openpyxl
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Dataset Fingerprint"
        ws.append(DETAILED_HEADERS)
        for r in rows:
            ws.append([r.get(k, "") for k in DETAILED_HEADERS])
        wb.save(str(out_path.with_suffix(".xlsx")))
        logger.info(f"Report saved to {out_path.with_suffix('.xlsx')}")
    except ImportError:
        csv_path = out_path.with_suffix(".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=DETAILED_HEADERS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in DETAILED_HEADERS})
        logger.info(f"Report saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Measure object size/shape/foreground-fraction stats directly from mask volumes.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory to search for mask volumes")
    parser.add_argument("--mask_name", type=str, default="images_mask", help="Mask folder name to match (e.g. 'images_mask' or 'Flatten_561_mask')")
    parser.add_argument("--output_name", type=str, default="dataset_fingerprint", help="Output filename (without extension)")
    parser.add_argument("--min_size", type=int, default=0, help="Drop connected components smaller than this many voxels (noise filtering)")
    parser.add_argument("--spacing_um", type=float, nargs=3, default=None, metavar=("Z", "Y", "X"), help="Physical voxel spacing in microns, to also report diameters in µm")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    mask_dirs = discover_mask_volumes(root, args.mask_name)
    logger.info(f"Found {len(mask_dirs)} mask volumes under {root}.")

    rows = []
    for msk_dir in tqdm(mask_dirs, desc="Fingerprinting volumes"):
        try:
            mask = FileReader(msk_dir).read(out_dtype=np.float32)
            fp = fingerprint_volume(mask, spacing_um=args.spacing_um, min_size=args.min_size)
            if fp is None:
                logger.warning(f"No objects found, skipping: {msk_dir}")
                continue
            rows.append({
                "parent_folder": msk_dir.parent.name,
                "volume_name": msk_dir.name,
                **fp,
            })
        except Exception as e:
            logger.warning(f"Failed on {msk_dir}: {e}")

    if not rows:
        logger.warning("No volumes could be fingerprinted.")
        return

    write_report(rows, root / args.output_name)

    diam_medians = [r["diameter_voxel_median"] for r in rows]
    families = [r["task_family_guess"] for r in rows]
    logger.info(
        f"Across {len(rows)} volumes: median object diameter {np.median(diam_medians):.1f} voxels "
        f"(range {np.min(diam_medians):.1f}-{np.max(diam_medians):.1f}). "
        f"task_family_guess counts: "
        f"blob={families.count('blob')}, tubular={families.count('tubular')}, unknown={families.count('unknown')}."
    )
    logger.info(
        "task_family_guess is a heuristic (median elongation threshold), not a substitute for a "
        "human check -- 改造計劃書.md's own open question #7 asks for a manual confirmation pass "
        "before trusting these numbers for config decisions."
    )


if __name__ == "__main__":
    main()
