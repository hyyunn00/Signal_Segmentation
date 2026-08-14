import numpy as np
import numba
from typing import Dict, Any, Optional

@numba.njit(parallel=True, nogil=True)
def _normalize_zscore_inplace_numba(data, mean, std):
    """Parallel in-place Z-score normalization using numba."""
    flat = data.ravel()
    n = flat.size
    eps = 1e-8
    inv_std = 1.0 / (std + eps)
    for i in numba.prange(n):
        flat[i] = (flat[i] - mean) * inv_std

@numba.njit(parallel=True, nogil=True)
def _normalize_minmax_inplace_numba(data, min_v, max_v):
    """Parallel in-place Min-Max normalization using numba."""
    flat = data.ravel()
    n = flat.size
    eps = 1e-8
    diff = max_v - min_v
    inv_diff = 1.0 / (diff + eps)
    for i in numba.prange(n):
        flat[i] = (flat[i] - min_v) * inv_diff

@numba.njit(parallel=True, nogil=True)
def _normalize_minmax_gamma_inplace_numba(data, min_v, max_v, gamma):
    """Parallel in-place Min-Max normalization followed by gamma correction."""
    flat = data.ravel()
    n = flat.size
    eps = 1e-8
    diff = max_v - min_v
    inv_diff = 1.0 / (diff + eps)
    inv_gamma = 1.0 / (gamma + eps)
    for i in numba.prange(n):
        v = (flat[i] - min_v) * inv_diff
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        flat[i] = v ** inv_gamma

@numba.njit(parallel=True, nogil=True)
def _normalize_percentile_inplace_numba(data, low_v, high_v):
    """Parallel in-place percentile clip + linear rescale to [0,1].

    Unlike z-score (mean/std of the whole volume, which for sparse
    fluorescence signal is dominated by background -- see 改造計劃書.md §6.2),
    this is anchored on the actual foreground/background intensity range via
    percentiles, so a handful of hot pixels or a small foreground fraction
    doesn't shift the normalization the way a mean/std would.
    """
    flat = data.ravel()
    n = flat.size
    eps = 1e-8
    diff = high_v - low_v
    inv_diff = 1.0 / (diff + eps)
    for i in numba.prange(n):
        v = (flat[i] - low_v) * inv_diff
        if v < 0.0:
            v = 0.0
        elif v > 1.0:
            v = 1.0
        flat[i] = v

@numba.njit(parallel=True, nogil=True)
def _apply_histogram_mapping_numba(data, lookup_table, range_min, range_max):
    """Parallel in-place histogram mapping using a lookup table and linear interpolation."""
    flat = data.ravel()
    n = flat.size
    
    num_bins = lookup_table.size
    bin_width = (range_max - range_min) / (num_bins - 1)
    inv_bin_width = 1.0 / (bin_width + 1e-12)

    for i in numba.prange(n):
        val = float(flat[i])
        
        # Clamp value to range
        if val <= range_min:
            flat[i] = lookup_table[0]
            continue
        if val >= range_max:
            flat[i] = lookup_table[num_bins - 1]
            continue
            
        # Linear interpolation within the lookup table
        idx_f = (val - range_min) * inv_bin_width
        idx_low = int(idx_f)
        idx_high = idx_low + 1
        
        if idx_high >= num_bins:
            flat[i] = lookup_table[num_bins - 1]
        else:
            # weight for high index
            w_high = idx_f - idx_low
            flat[i] = (1.0 - w_high) * lookup_table[idx_low] + w_high * lookup_table[idx_high]

def _percentile_value_from_cdf(cdf: np.ndarray, frac: float, r_min: float, r_max: float) -> float:
    """Inverts a binned CDF to find the intensity value at a given percentile
    fraction (0-1). r_min/r_max is the intensity domain the histogram bins
    span (see Normalizer's histogram-domain convention below)."""
    num_bins = cdf.size
    idx = int(np.searchsorted(cdf, frac))
    idx = min(max(idx, 0), num_bins - 1)
    bin_width = (r_max - r_min) / max(1, num_bins - 1)
    return r_min + idx * bin_width

class Normalizer:
    """
    Handles image intensity normalization based on global statistics.
    Supports Z-score, Min-Max, Percentile clipping, and Global Histogram Equalization.
    """
    def __init__(self, config: Dict[str, Any], volume_stats: Dict[str, Any]):
        """
        Args:
            config: Configuration dictionary containing both registry-level and
                    preprocess-level parameters.
            volume_stats: Dictionary containing 'mean', 'std', 'min', 'max', and optionally 'histogram'.
        """
        self.method = config.get("normalize_mode", "z-score").lower()

        # Z-score params
        self.mean = volume_stats.get("mean", 0.0)
        self.std = volume_stats.get("std", 1.0)
        self.std_multiplier = config.get("std_multiplier", 1.0)
        self.effective_std = self.std * self.std_multiplier

        # Min-Max params
        self.min_v = volume_stats.get("min", 0.0)
        self.max_v = volume_stats.get("max", 1.0)
        self.gamma = float(config.get("gamma", 1.0))

        # Histogram params (also feeds percentile mode below, which needs the
        # same binned distribution to find percentile cut points cheaply --
        # sorting the full volume just for two percentiles isn't worth it)
        self.bins = config.get("bins", 256)
        self.histogram = volume_stats.get("histogram")
        self.cdf = None

        if self.method in ("histogram", "percentile") and self.histogram is not None:
            self._prepare_histogram_cdf()

        # Percentile params (改造計劃書.md §6.2): per-volume percentile clip,
        # then linear rescale to [0,1]. Robust to sparse fluorescence signal
        # where mean/std (z-score) is dominated by background, and to
        # foreground-fraction differences across markers that make z-score
        # outputs incomparable between markers -- the point of this mode is
        # specifically to make hub-init checkpoints reusable across markers.
        self.percentiles = config.get("percentiles", [0.5, 99.5])
        self.percentile_low_v = self.min_v
        self.percentile_high_v = self.max_v
        if self.method == "percentile" and self.cdf is not None:
            r_min, r_max = self.min_v, self.max_v
            if r_min == r_max: r_max = r_min + 1.0
            p_low, p_high = self.percentiles
            self.percentile_low_v = _percentile_value_from_cdf(self.cdf, p_low / 100.0, r_min, r_max)
            self.percentile_high_v = _percentile_value_from_cdf(self.cdf, p_high / 100.0, r_min, r_max)
            if self.percentile_high_v <= self.percentile_low_v:
                self.percentile_high_v = self.percentile_low_v + 1.0

    def _prepare_histogram_cdf(self):
        """Pre-computes the CDF for histogram equalization / percentile lookup."""
        cdf_sum = self.histogram.cumsum()
        total_pixels = cdf_sum[-1]

        if total_pixels == 0:
            # Fallback to identity mapping if histogram is empty
            self.cdf = np.linspace(0, 1, self.bins).astype(np.float32)
        else:
            self.cdf = (cdf_sum / total_pixels).astype(np.float32)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization to the input data in-place or return a new array."""
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)

        if self.method == "z-score":
            _normalize_zscore_inplace_numba(data, self.mean, self.effective_std)
        elif self.method == "min-max":
            _normalize_minmax_inplace_numba(data, self.min_v, self.max_v)
        elif self.method == "min-max-gamma":
            _normalize_minmax_gamma_inplace_numba(data, self.min_v, self.max_v, self.gamma)
        elif self.method == "percentile":
            _normalize_percentile_inplace_numba(data, self.percentile_low_v, self.percentile_high_v)
        elif self.method == "histogram":
            if self.cdf is not None:
                # Use actual volume range for mapping
                r_min, r_max = self.min_v, self.max_v
                if r_min == r_max: r_max = r_min + 1.0
                _apply_histogram_mapping_numba(data, self.cdf, r_min, r_max)

        return data

    def get_background_value(self) -> float:
        """Returns the normalized equivalent of raw 0.0 for padding."""
        if self.method == "z-score":
            eps = 1e-8
            return (0.0 - self.mean) / (self.effective_std + eps)
        elif self.method == "percentile":
            eps = 1e-8
            v = (0.0 - self.percentile_low_v) / (self.percentile_high_v - self.percentile_low_v + eps)
            return max(0.0, min(1.0, v))
        elif self.method in ("min-max", "min-max-gamma"):
            eps = 1e-8
            v = (0.0 - self.min_v) / (self.max_v - self.min_v + eps)
            v = max(0.0, min(1.0, v))
            if self.method == "min-max-gamma":
                v = v ** (1.0 / (self.gamma + eps))
            return v
        elif self.method == "histogram":
            if self.cdf is not None:
                return float(self.cdf[0])
        return 0.0

def build_normalizer_from_config(full_config: Dict[str, Any], reader: Any, mode: str = "train") -> Normalizer:
    """
    Builds a Normalizer instance by merging registry parameters and local preprocess settings.
    
    Args:
        full_config: The complete project configuration.
        reader: A FileReader instance containing volume statistics.
        mode: The section to look for preprocess settings ('train' or 'inference').
    """
    registry = full_config.get("normalization", {})
    mode_config = full_config.get(mode, {})
    preprocess_config = mode_config.get("preprocess", {})
    
    method = preprocess_config.get("normalize_mode", "z-score")
    
    # Merge parameters: Registry (global defaults) + Preprocess (local overrides)
    method_defaults = registry.get(method, {})
    
    final_config = {**method_defaults, **preprocess_config}
    
    volume_stats = {
        "mean": reader.volume_mean,
        "std": reader.volume_std,
        "min": reader.volume_min,
        "max": reader.volume_max,
        "histogram": reader.volume_histogram
    }
    
    return Normalizer(final_config, volume_stats)
