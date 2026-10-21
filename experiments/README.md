# Paper experiments

All runners load `config.yaml` and accept `--config` plus a repository-relative
`--output` override. Run each experiment in a fresh output directory. No plotting
is invoked by these commands.

| Paper experiment | Command | Main output |
|---|---|---|
| RQ1: TECAD, Local, Concat | `python -m experiments.rq1` | `runs/rq1/METRICS.csv` |
| RQ1: native XCW rules | `python -m experiments.baselines.xcw` | `runs/xcw/predictions.jsonl` |
| RQ2: evidence, representation, fusion | `python -m experiments.rq2` | `runs/rq2/METRICS.csv` |
| RQ3: all 16 H/K configurations | `python -m experiments.rq3` | `runs/rq3/METRICS.csv` |
| RQ4: prepared descriptions | `python -m experiments.rq4 --mode B` | `runs/rq4/TIMING.csv` |
| RQ4: normalized records | `python -m experiments.rq4 --mode A-N --output runs/rq4-records` | `runs/rq4-records/TIMING.csv` |

RQ1/RQ2 report the main 0.005 calibration budget and the existing 0.0075/0.01
checks. RQ3 varies `--minutes 30 60 90 180` and `--k 1 2 5 10`, recalibrating each
configuration with the same frozen split. RQ4 runs Local, Top1, DP and TECAD.

The shared library fits reference preprocessing on reference normals, learns
fusion parameters on scale normals, and calibrates thresholds on calibration
normals. Evaluation labels are used only for metrics at these fixed choices.
Run metadata records the pinned inputs, command configuration and environment.
Model and per-target outputs are written under the chosen run directory.

For cloud runs, install the same package/environment and point a config override
to mounted inputs and a persistent output directory. No personal server name,
SSH credential, scheduler or absolute historical run path is embedded in code.
