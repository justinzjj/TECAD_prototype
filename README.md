# TECAD

Target-centred anomaly detection with local execution, candidate differences,
and candidate-history evidence. This repository contains the final Mean-only
prototype and the experiments supporting the paper.

## Quick start

Use Python 3.11–3.12 (local validation uses 3.12):

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,plots]'
```

Prepare the V5 package and the final paper feature revision as described in
[data/README.md](data/README.md). Data are not distributed through Git.

```sh
python -m tecad fit
python -m tecad detect
python -m experiments.rq1
pytest
```

The fitted model is saved under `runs/detector/`; detection writes
`runs/detections.csv`. The default paper operating point is a 90-minute window,
10 candidates, 24-hour/20-event history, 10th-neighbour distance, and a 0.5%
normal calibration budget. Expected evaluation counts are TECAD 180 TP / 23 FP,
Local 155 / 17 and Concat 84 / 17. See [experiments/README.md](experiments/README.md)
for all paper experiments and [plots/README.md](plots/README.md) for plotting.

## Layout and paths

- `src/tecad/`: shared data loading, evidence construction, fitting and detection.
- `experiments/`: experiment runners and the native XCW baseline adapter.
- `plots/`: figures generated from result CSVs.
- `data/`: ignored datasets plus the checked-in input manifest and instructions.
- `results/`: small frozen paper reference metrics.
- `runs/`: ignored generated models, predictions, timings and figures.

`config.yaml` controls all storage paths. Relative paths, including CLI output
paths, resolve from the repository root, independent of the working directory.
Use a local override without editing source code:

```yaml
# config.local.yaml (ignored by Git)
paths:
  dataset: /datasets/dataset-V5
  paper_features: /datasets/paper-features
  runs: runs/my-experiment
```

Pass `--config config.local.yaml` to an entry point. Existing experiment outputs
are protected; use another output directory for a new run. The default dataset
path can be a directory or a symlink.

Licensing and public dataset distribution must be finalized before publication.
No license for third-party rules or data is implied by this repository.
