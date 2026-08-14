#!/usr/bin/env python3
"""
Annotation-source detection (改造計劃書.md §6.4).

Checks whether a mask's boundary is explainable purely by intensity
thresholding, by comparing the intensity distribution of voxels just inside
the mask boundary against voxels just outside it. If a simple `intensity > T`
rule could have produced the boundary, those two distributions barely
overlap (AUC close to 1.0) -- a threshold-derived mask draws its boundary
exactly where intensity crosses T, so "inner shell" and "outer shell" voxels
are cleanly separable by intensity alone. Manual annotation, which follows
shape/context rather than only intensity, gives a much less separable
boundary (AUC roughly 0.7-0.85, per 改造計劃書.md's own bands).

This matters because a threshold-derived mask teaches a model "brightness
above some value" -- the single rule most likely to break the moment laser
power, antibody batch, or exposure time changes. If a chunk of the labeled
data used to build a shared_init hub turns out to be threshold-derived, it's
worth downweighting or excluding it before committing to a hub version
(改造計劃書.md §5.2 versioning).

This script only reads existing image/mask training-data folders; it never
modifies data or requires a trained model, so it can be run independently of
everything else in this repo.

Usage:
    python scripts/detect_annotation_source.py --data_root /path/to/training-data
    python scripts/detect_annotation_source.py --data_root ... --input_name Flatten_561 --mask_name Flatten_561_mask
"""
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from IO.reader import FileReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("detect_annotation_source")

DETAILED_HEADERS = [
    "parent_folder", "volume_name", "n_inner_shell_voxels", "n_outer_shell_voxels",
    "threshold_signature_auc", "annotation_guess",
]


def threshold_signature(image: np.ndarray, mask: np.ndarray, shell: int = 1) -> float:
    """Boundary-shell AUC, exactly as specified in 改造計劃書.md §6.4:
    ~1.00 => boundary fully explained by a single intensity threshold
             (mask is likely threshold-derived)
    ~0.7-0.85 => boundary follows shape/context, not just brightness
                 (mask is likely hand-annotated)
    """
    m = mask > 0.5
    inner = m & ~ndimage.binary_erosion(m, iterations=shell)
    outer = ndimage.binary_dilation(m, iterations=shell) & ~m
    vals = np.concatenate([image[inner], image[outer]])
    lab = np.concatenate([np.ones(int(inner.sum())), np.zeros(int(outer.sum()))])
    return roc_auc_score(lab, vals), int(inner.sum()), int(outer.sum())


def classify(auc: float) -> str:
    if auc >= 0.95:
        return "likely_thresholded"
    if 0.65 <= auc <= 0.90:
        return "likely_manual"
    return "ambiguous"


def discover_volumes(root: Path, input_name: str, mask_name: str) -> List[Tuple[Path, Path]]:
    """Finds (image_dir, mask_dir) pairs, mirroring the discovery rule
    IO/datasets.py::TrainMicroscopyDataset.from_folders() uses for training
    data (a dir named input_name with a sibling dir named mask_name)."""
    pairs = []
    for p in sorted(root.rglob("*")):
        if p.is_dir() and p.name == input_name:
            m_path = p.parent / mask_name
            if m_path.exists() and m_path.is_dir():
                pairs.append((p, m_path))
    return pairs


def write_report(rows: List[dict], out_path: Path):
    try:
        import openpyxl
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Annotation Source"
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
    parser = argparse.ArgumentParser(description="Check whether masks look threshold-derived vs hand-annotated (boundary-shell intensity AUC).")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory to search for image/mask volume pairs")
    parser.add_argument("--input_name", type=str, default="images", help="Image folder name to match (e.g. 'images' or 'Flatten_561')")
    parser.add_argument("--mask_name", type=str, default="images_mask", help="Mask folder name to match (e.g. 'images_mask' or 'Flatten_561_mask')")
    parser.add_argument("--output_name", type=str, default="annotation_source_report", help="Output filename (without extension)")
    parser.add_argument("--shell", type=int, default=1, help="Shell thickness in voxels for the boundary-erosion/dilation check")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    pairs = discover_volumes(root, args.input_name, args.mask_name)
    logger.info(f"Found {len(pairs)} image/mask volume pairs under {root}.")

    rows = []
    for img_dir, msk_dir in tqdm(pairs, desc="Checking volumes"):
        try:
            image = FileReader(img_dir).read(out_dtype=np.float32)
            mask = FileReader(msk_dir).read(out_dtype=np.float32)
            if image.shape != mask.shape:
                logger.warning(f"Shape mismatch, skipping: {img_dir} {image.shape} vs {msk_dir} {mask.shape}")
                continue
            if not np.any(mask > 0.5):
                logger.warning(f"Empty mask, skipping: {msk_dir}")
                continue

            auc, n_inner, n_outer = threshold_signature(image, mask, shell=args.shell)
            rows.append({
                "parent_folder": img_dir.parent.name,
                "volume_name": img_dir.name,
                "n_inner_shell_voxels": n_inner,
                "n_outer_shell_voxels": n_outer,
                "threshold_signature_auc": round(float(auc), 4),
                "annotation_guess": classify(auc),
            })
        except Exception as e:
            logger.warning(f"Failed on {img_dir}: {e}")

    if not rows:
        logger.warning("No volumes could be evaluated.")
        return

    write_report(rows, root / args.output_name)

    aucs = [r["threshold_signature_auc"] for r in rows]
    n_thresholded = sum(1 for r in rows if r["annotation_guess"] == "likely_thresholded")
    n_manual = sum(1 for r in rows if r["annotation_guess"] == "likely_manual")
    logger.info(
        f"AUC across {len(rows)} volumes: mean={np.mean(aucs):.3f}, median={np.median(aucs):.3f} -- "
        f"{n_thresholded} likely_thresholded, {n_manual} likely_manual, "
        f"{len(rows) - n_thresholded - n_manual} ambiguous."
    )
    if n_thresholded:
        logger.warning(
            f"{n_thresholded}/{len(rows)} volumes look threshold-derived (AUC>=0.95). Per "
            f"改造計劃書.md §6.4, consider excluding/downweighting these before using them to "
            f"build a shared_init hub -- they teach a rule ('brightness above T') that doesn't "
            f"transfer across laser power/antibody batch/exposure changes."
        )


if __name__ == "__main__":
    main()
