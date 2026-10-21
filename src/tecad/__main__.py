"""Fit a normal-only detector or score a V5 candidate scope with a saved detector."""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import load_config, ROOT
from .data import Dataset
from .model import Detector
from .scoring import calibrate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["fit", "detect"])
    p.add_argument("--config")
    p.add_argument("--model", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--minutes", type=int, default=90)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--beta", type=float, default=0.005)
    a = p.parse_args()
    c = load_config(a.config)
    data = Dataset(c["paths"]["dataset"], c["paths"].get("paper_features"))
    dest = a.model or c["paths"]["runs"] / "detector"
    dest = (dest if dest.is_absolute() else ROOT / dest).resolve()
    if a.command == "fit":
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError(f"Model directory is not empty: {dest}")
        raw, _ = data.build(a.minutes, a.k)
        m = Detector().fit(raw, data.rows)
        ns = len(data.rows["scale"])
        nc = len(data.rows["cal"])
        threshold, rank, _ = calibrate(m.scores()[ns : ns + nc], a.beta)
        m.save(dest)
        (dest / "protocol.json").write_text(
            json.dumps(
                dict(
                    minutes=a.minutes,
                    k=a.k,
                    beta=a.beta,
                    threshold=threshold,
                    cal_rank=rank,
                )
            )
        )
        print(f"Saved fitted detector: {dest}")
    else:
        protocol = json.loads((dest / "protocol.json").read_text())
        raw, _ = data.build(protocol["minutes"], protocol["k"])
        m = Detector.load(dest)
        scores = m.infer(raw)
        result = data.table[["target_id", "role"]].copy()
        result["score"] = scores
        result["prediction"] = (scores > protocol["threshold"]).astype(int)
        output = a.output or c["paths"]["runs"] / "detections.csv"
        output = (output if output.is_absolute() else ROOT / output).resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(output)


if __name__ == "__main__":
    main()
