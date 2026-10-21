"""Reference-only standardization and exact kth-neighbor distances."""

import math
import warnings
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


def fit_standardizer(x: np.ndarray) -> dict[str, np.ndarray]:
    """Fit StandardScaler-compatible moments while preserving missing-value semantics."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("x must be a nonempty matrix")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scaler = StandardScaler().fit(x)
    count = np.asarray(scaler.n_samples_seen_, dtype=np.float64)
    if count.ndim == 0:
        count = np.full(x.shape[1], float(count))
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    var = np.asarray(scaler.var_, dtype=np.float64)
    missing = count == 0
    mean = np.where(missing | ~np.isfinite(mean), 0.0, mean)
    var = np.where(missing | ~np.isfinite(var), 0.0, var)
    eps = np.finfo(np.float64).eps
    upper_bound = count * eps * var + (count * mean * eps) ** 2
    constant_mask = missing | (var <= upper_bound)
    scale = np.sqrt(var)
    scale[constant_mask] = 1.0
    return {
        "mean": mean,
        "var": var,
        "scale": scale,
        "count": count,
        "constant_mask": constant_mask,
        "upper_bound": upper_bound,
    }


def transform(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = np.divide(x - model["mean"], model["scale"])
    z = np.where(np.isfinite(x), z, 0.0)
    return np.ascontiguousarray(z / math.sqrt(x.shape[1]))


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    top = float(np.max(x)) if len(x) else 0.0
    return float(top * np.sqrt(np.mean((x / top) ** 2))) if top > 0 else 0.0


def kth_distance(
    x: np.ndarray, reference: np.ndarray, neighbors: int = 10, chunk: int = 128
) -> np.ndarray:
    if len(reference) < neighbors:
        raise ValueError("insufficient reference")
    out = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), chunk):
        squared = cdist(x[start : start + chunk], reference, metric="sqeuclidean")
        out[start : start + chunk] = np.sqrt(
            np.partition(squared, neighbors - 1, axis=1)[:, neighbors - 1]
        )
    return out
