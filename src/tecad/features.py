"""Versioned P16 execution differences with target log-only reconstruction."""

import numpy as np

VERSION = "JSTAT_P_LOG_ONLY_V1"
FIELDS = (
    "success",
    "native_value_log",
    "gas_log",
    "log_only_count_log",
    "transfer_count_log",
    "mint_ratio",
    "burn_ratio",
    "self_ratio",
)


def pair_features(target, candidate, target_valid, candidate_valid):
    d = np.asarray(target, dtype=np.float64).copy()
    s = np.asarray(candidate, dtype=np.float64)
    dm = np.asarray(target_valid, dtype=bool).copy()
    sm = np.asarray(candidate_valid, dtype=bool)
    if d.ndim != 2 or d.shape[1] != 8 or any(a.shape != d.shape for a in (s, dm, sm)):
        raise ValueError("expected matching N x 8 operands")
    # D[3] counts logs + movements; D[4] counts movements.
    valid = dm[:, 3] & dm[:, 4] & np.isfinite(d[:, 3]) & np.isfinite(d[:, 4])
    with np.errstate(over="ignore", invalid="ignore"):
        counts = np.expm1(d[valid][:, [3, 4]])
    rounded = np.rint(counts)
    if not np.isfinite(counts).all() or not np.allclose(
        counts, rounded, rtol=1e-7, atol=1e-5
    ):
        raise ValueError("noninteger_reconstructed_event_count")
    if (rounded < 0).any() or (rounded[:, 0] < rounded[:, 1]).any():
        raise ValueError("negative_reconstructed_count")
    d[:, 3] = 0
    d[valid, 3] = np.log1p(rounded[:, 0] - rounded[:, 1])
    dm[:, 3] = valid
    observed = dm & sm & np.isfinite(d) & np.isfinite(s)
    diff = d - s
    p = np.stack((diff, np.abs(diff)), axis=-1).reshape(-1, 16)
    mask = np.repeat(observed, 2, axis=1)
    mask[:, 2:4] = False
    return np.where(mask, p, 0), mask


def select_members(pool, seconds, k):
    if seconds <= 0 or k < 1:
        raise ValueError("invalid candidate scope")
    q = pool.loc[(pool.gap_seconds > 0) & (pool.gap_seconds <= seconds)].sort_values(
        ["target_id", "gap_seconds", "candidate_key"], kind="mergesort"
    )
    if q.duplicated(["target_id", "candidate_key"]).any():
        raise ValueError("duplicate candidate identity")
    q = q.groupby("target_id", sort=False).head(k).copy()
    q["rank"] = q.groupby("target_id", sort=False).cumcount() + 1
    return q


def mean_valid(values, mask, indices, n):
    values = np.asarray(values, float)
    mask = np.asarray(mask, bool)
    indices = np.asarray(indices, int)
    if values.shape != mask.shape or len(indices) != len(values):
        raise ValueError("invalid shapes")
    if np.any(mask & ~np.isfinite(values)):
        raise ValueError("nonfinite valid value")
    sums = np.zeros((n, values.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, indices, np.where(mask, values, 0))
    np.add.at(counts, indices, mask)
    return np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)


def evidence_arrays(values, masks, width, states):
    out = np.zeros((len(values), width))
    valid = np.zeros_like(out, dtype=bool)
    for i, (v, m, state) in enumerate(zip(values, masks, states)):
        if state == "not_materialized":
            continue
        if np.shape(v) != (width,) or np.shape(m) != (width,):
            raise ValueError("Malformed observed evidence")
        out[i] = v
        valid[i] = m
    assert np.isfinite(out[valid]).all()
    return out, valid


def aggregate(values, mask, target_index, n):
    v = np.asarray(values, float)
    m = np.asarray(mask, bool)
    ix = np.asarray(target_index, int)
    if (
        v.ndim != 2
        or m.shape != v.shape
        or len(ix) != len(v)
        or (ix < 0).any()
        or (ix >= n).any()
        or (not np.isfinite(v[m]).all())
    ):
        raise ValueError("Invalid candidate aggregation inputs")
    count = np.zeros((n, v.shape[1]), int)
    total = np.zeros_like(count, float)
    np.add.at(count, ix, m)
    np.add.at(total, ix, np.where(m, v, 0))
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    ss = np.zeros_like(total)
    np.add.at(ss, ix, np.where(m, (v - mean[ix]) ** 2, 0))
    std = np.sqrt(np.divide(ss, count, out=np.full_like(ss, np.nan), where=count > 0))
    return np.column_stack([mean, std])
