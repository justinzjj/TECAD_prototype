"""Opt-in numerical regression from source evidence, never historical scores."""

import os
import numpy as np
import pytest
from tecad.config import load_config
from tecad.data import Dataset
from tecad.model import Detector
from tecad.scoring import calibrate


@pytest.mark.skipif(
    os.environ.get("TECAD_V5_INTEGRATION") != "1",
    reason="set TECAD_V5_INTEGRATION=1 with local V5 and paper features",
)
def test_fresh_v5_primary_and_model_roundtrip(tmp_path):
    config = load_config()
    d = Dataset(config["paths"]["dataset"], config["paths"]["paper_features"])
    raw, _ = d.build()
    m = Detector().fit(raw, d.rows)
    ns, nc = len(d.rows["scale"]), len(d.rows["cal"])
    scores = m.scores()
    cut, _, _ = calibrate(scores[ns : ns + nc], 0.005)
    y = d.table.iloc[d.rows["test"]].dataset_label.to_numpy(int)
    p = scores[ns + nc :] > cut
    assert (int(((y == 1) & p).sum()), int(((y == 0) & p).sum())) == (180, 23)
    m.save(tmp_path)
    loaded = Detector.load(tmp_path)
    q = {b: x[d.rows["test"]] for b, x in raw.items()}
    np.testing.assert_allclose(loaded.infer(q), scores[ns + nc :], rtol=0, atol=0)
