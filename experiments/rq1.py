"""RQ1: freshly fit TECAD, local, concatenation."""

import numpy as np
from tecad.data import BLOCKS
from tecad.distance import transform, kth_distance, rms
from tecad.model import Detector
from .common import arguments, setup, write_results, parameters


def main():
    args = arguments(__doc__).parse_args()
    config, out, data = setup(args, "rq1")
    raw, _ = data.build()
    model = Detector().fit(raw, data.rows)
    parameters(model, out)
    scores = {"TECAD": model.scores(), "Local": model.scores((0,))}
    joint = np.column_stack([transform(model.models[b], raw[b]) for b in BLOCKS])
    ref = joint[data.rows["reference"]]
    q = joint[model.query]
    ns = len(data.rows["scale"])
    d = kth_distance(q, ref, 10)
    scores["Concat"] = d / rms(d[:ns])
    print(write_results(data, scores, out).to_string(index=False))


if __name__ == "__main__":
    main()
