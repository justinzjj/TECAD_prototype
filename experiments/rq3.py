"""RQ3: candidate time-window and candidate-budget sensitivity grid."""

import pandas as pd
from tecad.model import Detector
from tecad.scoring import learn_rates, combine
from .common import arguments, setup, write_results, parameters


def main():
    parser = arguments(__doc__)
    parser.add_argument("--minutes", type=int, nargs="+", default=[30, 60, 90, 180])
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 5, 10])
    args = parser.parse_args()
    config, out, data = setup(args, "rq3")
    all_metrics = []
    for h in args.minutes:
        for k in args.k:
            name = f"H{h}_K{k}"
            dest = out / name
            dest.mkdir(exist_ok=True)
            raw, membership = data.build(h, k)
            m = Detector().fit(raw, data.rows)
            parameters(m, dest)
            raw_rates, _ = learn_rates(
                m.distances[: len(data.rows["scale"])],
                m.support[: len(data.rows["scale"])],
            )
            score = combine(m.distances, raw_rates)
            pd.DataFrame(
                {
                    "block": ["D", "P", "Ctx"],
                    "lambda_raw": raw_rates,
                    "weight": raw_rates / raw_rates.sum(),
                }
            ).to_csv(dest / "FUSION.csv", index=False)
            metrics = write_results(data, {name: score}, dest, betas=(0.005,))
            metrics["H_minutes"] = h
            metrics["K"] = k
            membership.to_parquet(dest / "MEMBERSHIP.parquet", index=False)
            all_metrics.append(metrics)
            print(metrics.to_string(index=False), flush=True)
    pd.concat(all_metrics, ignore_index=True).to_csv(out / "METRICS.csv", index=False)


if __name__ == "__main__":
    main()
