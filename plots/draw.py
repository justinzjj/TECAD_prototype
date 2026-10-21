"""Draw paper ablation and candidate-scope panels from experiment CSVs."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tecad.config import load_config, ROOT


def ablation(frame):
    if "beta" in frame:
        frame = frame[np.isclose(frame.beta, 0.005)]
    if frame.arm.duplicated().any():
        raise ValueError("Duplicate method rows")
    frame = frame.set_index("arm")
    groups = [
        (
            ["D", "DP", "DC", "TECAD"],
            ["Local (D)", "D+P", "D+C", "TECAD"],
            "Evidence composition",
        ),
        (
            ["TECAD", "Top1", "MeanStd", "D_state"],
            ["TECAD", "Top-1", "Mean+Std", "D+state"],
            "Candidate representation",
        ),
        (
            ["TECAD", "Equal_NoRMS", "Equal_RMS", "Concat"],
            ["TECAD", "Equal", "Equal+RMS", "Concat"],
            "Scoring and fusion",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), layout="constrained")
    for col, (keys, labels, title) in enumerate(groups):
        part = frame.loc[keys]
        colors = [
            "#0072B2" if k == "TECAD" else c
            for k, c in zip(keys, ["#666666", "#D55E00", "#009E73", "#AA4499"])
        ]
        ax = axes[0, col]
        for row, label, color in zip(part.itertuples(), labels, colors):
            ax.scatter(row.FP, row.TP, color=color, s=35)
            offset = (
                (-34, 10)
                if label == "TECAD"
                else (4, -14) if label in ("D+P", "Top-1") else (4, 6)
            )
            ax.annotate(
                label,
                (row.FP, row.TP),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
            )
        ax.set(xlabel="False positives", ylabel="True positives", title=title)
        ax.margins(0.25)
        ax.grid(alpha=0.15)
        ax = axes[1, col]
        ax.barh(labels, part.F1 * 100, color=colors)
        for i, value in enumerate(part.F1 * 100):
            ax.text(value + 1, i, f"{value:.2f}", va="center", fontsize=8)
        ax.invert_yaxis()
        ax.set(xlim=(0, 105), xlabel="F1 (%)")
    return fig


def scope(frame):
    if "H_minutes" not in frame:
        frame = frame.copy()
        frame["H_minutes"] = frame.arm.str.extract(r"H(\d+)").astype(int)
        frame["K"] = frame.arm.str.extract(r"_K(\d+)").astype(int)
    if frame.duplicated(["H_minutes", "K"]).any() or len(frame) != 16:
        raise ValueError("Expected exactly one row for each of 16 scope configurations")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), layout="constrained")
    for ax, key in zip(axes, ["Recall", "FPR", "F1", "AP"]):
        values = (
            frame.pivot(index="H_minutes", columns="K", values=key)
            .reindex(index=[30, 60, 90, 180], columns=[1, 2, 5, 10])
            .to_numpy()
            * 100
        )
        if not np.isfinite(values).all():
            raise ValueError("Incomplete grid")
        ax.imshow(values, cmap="Blues", aspect="auto")
        for i in range(4):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{values[i,j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=(
                        "white"
                        if values[i, j] > (values.min() + values.max()) / 2
                        else "black"
                    ),
                )
        ax.set(
            xticks=range(4),
            xticklabels=[1, 2, 5, 10],
            yticks=range(4),
            yticklabels=[30, 60, 90, 180],
            xlabel="K",
            ylabel="H (min)",
            title=f"{key} (%)",
        )
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("figure", choices=["ablation", "scope"])
    p.add_argument("--config")
    p.add_argument(
        "--reference",
        action="store_true",
        help="Plot saved paper metrics instead of a new run",
    )
    p.add_argument("--input", type=Path)
    args = p.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    if args.input:
        source = args.input if args.input.is_absolute() else ROOT / args.input
    elif args.reference:
        source = paths["reference_results"] / (
            "reference_metrics.csv"
            if args.figure == "ablation"
            else "reference_scope.csv"
        )
    else:
        source = (
            paths["runs"]
            / ("rq2" if args.figure == "ablation" else "rq3")
            / "METRICS.csv"
        )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure = globals()[args.figure](pd.read_csv(source))
    paths["figures"].mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        figure.savefig(paths["figures"] / f"{args.figure}.{ext}", dpi=220)
    plt.close(figure)
    print(f'Figures written to {paths["figures"]}')


if __name__ == "__main__":
    main()
