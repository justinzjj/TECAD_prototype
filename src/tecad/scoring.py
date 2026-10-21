"""Pure scoring rules fixed before reading evaluation labels."""

import math

import numpy as np


def calibrate(scores, beta):
    values = np.asarray(scores, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("cal scores must be a nonempty finite vector")
    rank = math.ceil((len(values) + 1) * (1 - beta))
    if not 1 <= rank <= len(values):
        raise ValueError("invalid rank")
    cut = float(np.sort(values)[rank - 1])
    return cut, rank, int(np.sum(values > cut))


def learn_rates(u, support):
    values = np.asarray(u, float)
    mask = np.asarray(support, bool)
    if values.ndim != 2 or mask.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("incompatible scale scores and support")
    rates = []
    counts = []
    for j in range(values.shape[1]):
        positive = values[:, j][mask[:, j] & (values[:, j] > 0)]
        if (
            not len(positive)
            or not np.isfinite(positive.mean())
            or positive.mean() <= 0
        ):
            raise ValueError(f"block {j}: no estimable strict positive scale rate")
        rates.append(1 / positive.mean())
        counts.append(len(positive))
    return np.array(rates), np.array(counts)


def combine(u, rates, active=None, equal=False):
    values = np.asarray(u, float)
    rates = np.asarray(rates, float)
    if values.ndim != 2 or values.shape[1] != len(rates):
        raise ValueError("score width does not match rates")
    if not np.isfinite(rates).all() or not np.all(rates > 0):
        raise ValueError("rates must be finite and positive")
    if active is None:
        active = tuple(range(len(rates)))
    if not active:
        raise ValueError("no active blocks")
    w = np.zeros(len(rates))
    indices = list(active)
    selected = np.ones(len(indices)) if equal else rates[indices]
    w[indices] = selected / selected.sum()
    return values @ w


def paired(y, before, after):
    y = np.asarray(y, int)
    before = np.asarray(before, bool)
    after = np.asarray(after, bool)
    if y.shape != before.shape or y.shape != after.shape:
        raise ValueError("paired arrays are not aligned")
    return dict(
        new_TP=int(np.sum((y == 1) & ~before & after)),
        lost_TP=int(np.sum((y == 1) & before & ~after)),
        new_FP=int(np.sum((y == 0) & ~before & after)),
        removed_FP=int(np.sum((y == 0) & before & ~after)),
    )


def outer_std_zero_when_finite(std):
    """Top1 retains NaN for missing evidence; supported std coordinates are zero."""
    values = np.asarray(std, float)
    return bool(np.all(values[np.isfinite(values)] == 0))
