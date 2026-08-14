#!/usr/bin/env python3
"""
3D-aware evaluation (P0 fix for 改造計劃書.md §3.4).

The existing analysis.py evaluates each z-slice .tif file independently and
labels connected components per-slice, so a single 3D object spanning
several z-slices gets counted once per slice it touches (gt_count/pr_count
inflate several-fold; see the --compare-2d diagnostic below).

This script instead stacks every slice in a GT/prediction folder pair into a
single (Z, Y, X) volume, runs 26-connectivity connected-component labeling
once on the full volume, and matches objects by centroid distance (tolerance
= that object's own equivalent radius) instead of per-slice IoU.

This is intentionally a separate, additive script: it does not modify
train.py or analysis.py's existing evaluate_triplet() behavior, so it can be
run against existing model outputs to measure the 2D-vs-3D counting gap
before touching anything else in the pipeline.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
from scipy.spatial import cKDTree
from skimage.measure import label, regionprops
from tqdm import tqdm

from analysis import discover_triplets, evaluate_triplet, list_files, to_binary, write_results
from utils.metrics import accumulate_confusion, compute_cldice, compute_metrics, compute_pr_auc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("analysis3d")

DETAILED_HEADERS_3D = [
    "parent_folder", "image_folder", "model_name", "files_evaluated", "voxels_total",
    "tp", "fp", "fn", "tn", "accuracy", "precision", "recall", "f1", "mcc", "hausdorff", "cldice", "pr_auc",
    "obj_f1_3d", "obj_precision_3d", "obj_recall_3d", "gt_count_3d", "pr_count_3d",
    "gt_count_2d", "pr_count_2d", "count_inflation_ratio",
]

SUMMARY_HEADERS_3D = [
    "model_name",
    "accuracy_mean", "accuracy_std", "precision_mean", "precision_std",
    "recall_mean", "recall_std", "f1_mean", "f1_std", "mcc_mean",
    "hausdorff_mean", "cldice_mean", "obj_f1_3d_mean", "pr_auc_mean",
    "count_inflation_ratio_mean", "count_inflation_ratio_std",
]

METRICS_TO_AGG = ["accuracy", "precision", "recall", "f1", "mcc", "hausdorff", "cldice", "obj_f1_3d", "pr_auc", "count_inflation_ratio"]


def _filter_small_labels(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Zeroes out connected components smaller than min_size and relabels."""
    if labels.max() == 0:
        return labels
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = True  # background stays mapped to 0
    remap = np.where(keep, np.arange(sizes.size), 0)
    filtered = remap[labels]
    return label(filtered > 0, connectivity=3)


def compute_3d_object_metrics(gt_vol: np.ndarray, pr_vol: np.ndarray, min_size: int = 0) -> Dict[str, float]:
    """Detection metrics on a full 3D volume: 26-connectivity labeling once,
    then greedy nearest-centroid matching (tolerance = the GT object's own
    equivalent radius) instead of per-slice IoU matching.
    """
    gt_labels = label(gt_vol, connectivity=3)
    pr_labels = label(pr_vol, connectivity=3)

    if min_size > 0:
        gt_labels = _filter_small_labels(gt_labels, min_size)
        pr_labels = _filter_small_labels(pr_labels, min_size)

    num_gt = int(gt_labels.max())
    num_pr = int(pr_labels.max())

    if num_gt == 0:
        return {"obj_f1_3d": 0.0, "obj_precision_3d": 0.0, "obj_recall_3d": 0.0, "gt_count_3d": 0, "pr_count_3d": num_pr}
    if num_pr == 0:
        return {"obj_f1_3d": 0.0, "obj_precision_3d": 0.0, "obj_recall_3d": 0.0, "gt_count_3d": num_gt, "pr_count_3d": 0}

    gt_regions = regionprops(gt_labels)
    pr_regions = regionprops(pr_labels)

    gt_centroids = np.array([r.centroid for r in gt_regions])
    pr_centroids = np.array([r.centroid for r in pr_regions])
    # regionprops' `.area` is the voxel count for an ND region; derive an
    # equivalent-sphere radius from it directly instead of relying on
    # equivalent_diameter attribute names that differ across skimage versions.
    gt_voxel_counts = np.array([r.area for r in gt_regions], dtype=np.float64)
    gt_radius = (3.0 * gt_voxel_counts / (4.0 * np.pi)) ** (1.0 / 3.0)

    tree = cKDTree(pr_centroids)
    matched_pr = set()
    tp = 0
    for gi in range(num_gt):
        candidate_idx = tree.query_ball_point(gt_centroids[gi], r=gt_radius[gi])
        if not candidate_idx:
            continue
        candidate_idx.sort(key=lambda pi: np.linalg.norm(gt_centroids[gi] - pr_centroids[pi]))
        for pi in candidate_idx:
            if pi not in matched_pr:
                matched_pr.add(pi)
                tp += 1
                break

    fp = num_pr - tp
    fn = num_gt - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

    return {
        "obj_f1_3d": float(f1),
        "obj_precision_3d": float(prec),
        "obj_recall_3d": float(rec),
        "gt_count_3d": int(num_gt),
        "pr_count_3d": int(num_pr),
    }


def _pair_and_stack(gt_dir: Path, pred_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Reads every GT slice + its matching prediction slice, stacks them into
    (Z, Y, X) volumes. Returns (g_vol_binary, p_vol_binary, g_vol_raw, p_vol_raw)
    or None if no matching, same-shape slices were found.
    """
    g_slices, p_slices, g_slices_raw, p_slices_raw = [], [], [], []
    for gtf in list_files(gt_dir):
        prf = pred_dir / gtf.name
        if not prf.exists():
            alt = gtf.stem + (".tiff" if gtf.suffix == ".tif" else ".tif")
            prf = pred_dir / alt
        if not prf.exists():
            continue
        try:
            g_raw = tifffile.imread(str(gtf))
            p_raw = tifffile.imread(str(prf))
            if g_raw.shape != p_raw.shape:
                continue
            g_slices_raw.append(g_raw)
            p_slices_raw.append(p_raw)
            g_slices.append(to_binary(g_raw))
            p_slices.append(to_binary(p_raw))
        except Exception as e:
            logger.warning(f"Error reading {gtf.name}: {e}")

    if not g_slices:
        return None

    return (
        np.stack(g_slices, axis=0),
        np.stack(p_slices, axis=0),
        np.stack(g_slices_raw, axis=0),
        np.stack(p_slices_raw, axis=0),
    )


def evaluate_triplet_3d(gt_dir: Path, pred_dir: Path, metric_cfg: dict = None, compare_2d: bool = False) -> Dict[str, float]:
    metric_cfg = metric_cfg or {}
    hd_p = metric_cfg.get("hausdorff", {}).get("hd_percentile", 95.0)
    min_size = metric_cfg.get("object_metrics", {}).get("min_size", 0)

    stacked = _pair_and_stack(gt_dir, pred_dir)
    if stacked is None:
        return {"files_evaluated": 0}
    g_vol, p_vol, g_vol_raw, p_vol_raw = stacked

    tp, fp, fn, tn = accumulate_confusion(0, 0, 0, 0, g_vol, p_vol)
    metrics = compute_metrics(tp, fp, fn, tn, g_vol, p_vol, hd_percentile=hd_p)
    metrics["files_evaluated"] = g_vol.shape[0]
    metrics["voxels_total"] = metrics.pop("total", 0.0)
    metrics["cldice"] = compute_cldice(g_vol, p_vol)
    metrics["pr_auc"] = compute_pr_auc(g_vol, p_vol_raw)

    metrics.update(compute_3d_object_metrics(g_vol, p_vol, min_size=min_size))

    if compare_2d:
        legacy = evaluate_triplet(gt_dir, pred_dir, metric_cfg=metric_cfg)
        gt_2d = legacy.get("gt_count", 0)
        pr_2d = legacy.get("pr_count", 0)
        gt_3d = metrics.get("gt_count_3d", 0)
        metrics["gt_count_2d"] = gt_2d
        metrics["pr_count_2d"] = pr_2d
        metrics["count_inflation_ratio"] = (gt_2d / gt_3d) if gt_3d else float("nan")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="3D-aware evaluation: connected-component labeling + centroid-distance object matching on the full volume, instead of per-slice 2D counting.")
    parser.add_argument("--base_dir", type=str, required=True, help="Root directory to search")
    parser.add_argument("--output_name", type=str, default="evaluation_report_3d", help="Output filename")
    parser.add_argument("--config", type=str, default=None, help="Path to config file for metric parameters")
    parser.add_argument("--compare-2d", action="store_true", help="Also compute the legacy per-slice 2D object counts (gt_count_2d/pr_count_2d) and the resulting count_inflation_ratio, for diagnosing the 2D-vs-3D counting gap.")
    args = parser.parse_args()

    metric_cfg = {}
    if args.config:
        try:
            with open(args.config, "r") as f:
                metric_cfg = json.load(f).get("metrics", {})
            logger.info(f"Loaded metric configuration from {args.config}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")

    root = Path(args.base_dir).resolve()
    triplets = discover_triplets(root)
    logger.info(f"Found {len(triplets)} GT/prediction pairs to analyze (3D mode, compare_2d={args.compare_2d}).")

    all_results: List[Dict[str, object]] = []
    for gt_dir, img_dir, model_name, p_dir in tqdm(triplets, desc="Evaluating (3D)"):
        metrics = evaluate_triplet_3d(gt_dir, p_dir, metric_cfg=metric_cfg, compare_2d=args.compare_2d)
        if metrics.get("files_evaluated", 0) > 0:
            all_results.append({
                "parent_folder": gt_dir.parent.name,
                "image_folder": img_dir.name,
                "model_name": model_name,
                **metrics,
            })

    if not all_results:
        logger.warning("No matching GT/Prediction pairs found."); return

    model_names = sorted(set(r["model_name"] for r in all_results))
    summary_rows = []
    for m_name in model_names:
        m_results = [r for r in all_results if r["model_name"] == m_name]
        row = {"model_name": m_name}
        for metric in METRICS_TO_AGG:
            values = [
                r[metric] for r in m_results
                if metric in r and r[metric] is not None and not (isinstance(r[metric], float) and np.isnan(r[metric]))
            ]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else 0.0
            row[f"{metric}_std"] = float(np.std(values)) if values else 0.0
        summary_rows.append(row)

    write_results(all_results, summary_rows, root / args.output_name, detailed_headers=DETAILED_HEADERS_3D, summary_headers=SUMMARY_HEADERS_3D)

    if args.compare_2d:
        ratios = [r["count_inflation_ratio"] for r in all_results if not np.isnan(r.get("count_inflation_ratio", float("nan")))]
        if ratios:
            logger.info(
                f"2D-vs-3D GT object count inflation: mean={np.mean(ratios):.2f}x, "
                f"median={np.median(ratios):.2f}x, max={np.max(ratios):.2f}x across {len(ratios)} volumes."
            )


if __name__ == "__main__":
    main()
