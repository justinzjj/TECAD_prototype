import numpy as np
import pandas as pd
from tecad.scoring import calibrate, combine, learn_rates
from tecad.features import mean_valid, select_members
from tecad.distance import fit_standardizer, transform, kth_distance


def test_strict_threshold_ties():
    cut, rank, alarms = calibrate(np.array([0.0, 1.0, 2.0, 2.0]), 0.4)
    assert (cut, rank, alarms) == (2.0, 3, 0)


def test_positive_supported_rate_and_scale_equivalence():
    raw = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 0.0]])
    support = np.array([[True, True], [True, True], [True, False]])
    scales = np.array([3.0, 2.0])
    rate, count = learn_rates(raw / scales, support)
    assert count.tolist() == [2, 2]
    factor = rate.sum() / (rate / scales).sum()
    np.testing.assert_allclose(
        combine(raw, rate / scales), factor * combine(raw / scales, rate)
    )


def test_scope_order_missing_and_coordinate_means():
    pool = pd.DataFrame(
        {
            "target_id": ["t"] * 4,
            "candidate_key": ["b", "a", "z", "q"],
            "gap_seconds": [2, 2, 0, 61],
        }
    )
    assert select_members(pool, 60, 1).candidate_key.tolist() == ["a"]
    x = mean_valid([[2.0, 0], [4.0, 8]], [[True, False], [True, True]], [0, 0], 2)
    np.testing.assert_equal(x, [[3.0, 8.0], [np.nan, np.nan]])


def test_standardization_and_neighbor_rank():
    x = np.array([[0.0, np.nan], [2.0, np.nan], [4.0, np.nan]])
    model = fit_standardizer(x)
    z = transform(model, x)
    assert np.isfinite(z).all() and np.all(z[:, 1] == 0)
    np.testing.assert_allclose(kth_distance(z[:1], z, 2), [np.sqrt(0.75)])
