"""RQ2: evidence, candidate representation and score fusion ablations."""

import numpy as np
from tecad.data import BLOCKS
from tecad.distance import transform, kth_distance, rms
from tecad.model import Detector
from .common import arguments, setup, write_results, parameters


def main():
    args = arguments(__doc__).parse_args()
    config, out, data = setup(args, "rq2")
    raw, _ = data.build()
    m = Detector().fit(raw, data.rows)
    parameters(m, out)
    scores = {
        "TECAD": m.scores(),
        "D": m.scores((0,)),
        "DP": m.scores((0, 1)),
        "DC": m.scores((0, 2)),
        "Equal_RMS": m.scores(equal=True),
        "Learned_NoRMS": m.scores(normalized=False),
        "Equal_NoRMS": m.scores(equal=True, normalized=False),
    }
    for name, k, representation in [
        ("Top1", 1, "mean"),
        ("MeanStd", 10, "mean_std"),
        ("D_state", 10, "state"),
    ]:
        x, _ = data.build(k=k, representation=representation)
        scores[name] = Detector().fit(x, data.rows).scores()
    joint = np.column_stack([transform(m.models[b], raw[b]) for b in BLOCKS])
    d = kth_distance(joint[m.query], joint[data.rows["reference"]], 10)
    scores["Concat"] = d / rms(d[: len(data.rows["scale"])])
    print(write_results(data, scores, out).to_string(index=False))


if __name__ == "__main__":
    main()
