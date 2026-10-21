import argparse
import json
import hashlib, platform, sys
import sklearn
import scipy,pyarrow
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from tecad.config import load_config, ROOT
from tecad.data import Dataset
from tecad.model import Detector
from tecad.scoring import calibrate


def arguments(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config")
    p.add_argument("--output", type=Path)
    return p


def setup(args, name):
    config = load_config(args.config)
    out = args.output or config["paths"]["runs"] / name
    out = out if out.is_absolute() else ROOT / out
    if any((out / name).exists() for name in ("METRICS.csv", "TIMING.csv", "RUN.json")):
        raise FileExistsError(f"Existing run: {out}; choose a fresh --output directory")
    data = Dataset(config["paths"]["dataset"], config["paths"].get("paper_features"))
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        experiment=name,
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        input_hashes=data.hashes,
        python=sys.version,
        platform=platform.platform(),
        numpy=np.__version__,
        pandas=pd.__version__,
        sklearn=sklearn.__version__,
        protocol=dict(
            neighbors=10,
            reference=8657,
            scale=1111,
            cal=1113,
            test=5191,
            decision="score > threshold",
        ),
    )
    (out / "RUN.json").write_text(json.dumps(metadata, indent=2))
    return config, out, data


def metric(y, s, p):
    tp = int(((y == 1) & p).sum())
    fp = int(((y == 0) & p).sum())
    tn = int(((y == 0) & ~p).sum())
    fn = int(((y == 1) & ~p).sum())
    return dict(
        TP=tp,
        FP=fp,
        TN=tn,
        FN=fn,
        Precision=tp / (tp + fp) if tp + fp else 0.0,
        Recall=tp / (tp + fn),
        F1=2 * tp / (2 * tp + fp + fn),
        FPR=fp / (fp + tn),
        AP=float(average_precision_score(y, s)),
    )


def write_results(dataset, scores, out, betas=(0.005, 0.0075, 0.01)):
    ns = len(dataset.rows["scale"])
    nc = len(dataset.rows["cal"])
    ev = dataset.table.iloc[dataset.rows["test"]].copy()
    cal = pd.DataFrame({"target_id": dataset.ids[dataset.rows["cal"]]})
    metrics = []
    y = ev.dataset_label.to_numpy(int)
    for arm, s in scores.items():
        c = s[ns : ns + nc]
        s = s[ns + nc :]
        cal[arm] = c
        ev[f"score_{arm}"] = s
        for beta in betas:
            cut, rank, alarms = calibrate(c, beta)
            p = s > cut
            ev[f"pred_{arm}_b{beta}"] = p.astype(int)
            metrics.append(
                dict(
                    arm=arm,
                    beta=beta,
                    threshold=cut,
                    cal_rank=rank,
                    cal_alarms=alarms,
                    **metric(y, s, p),
                )
            )
    pd.DataFrame(metrics).to_csv(out / "METRICS.csv", index=False, float_format="%.17g")
    ev.to_csv(out / "EVAL_SCORES_PREDICTIONS.csv", index=False, float_format="%.17g")
    cal.to_csv(out / "CAL_SCORES.csv", index=False, float_format="%.17g")
    return pd.DataFrame(metrics)


def parameters(model, out):
    pd.DataFrame(
        {
            "block": ["D", "P", "Ctx"],
            "scale_rms": model.scales,
            "lambda_rate": model.rates,
            "positive_scale_n": model.positive_counts,
            "weight": model.rates / model.rates.sum(),
        }
    ).to_csv(out / "PARAMETERS.csv", index=False)
    model.save(out / "model")
