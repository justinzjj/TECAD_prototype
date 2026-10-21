# Paper figures

Install plotting dependencies with `pip install -e '.[plots]'`.

```sh
python -m plots.draw ablation
python -m plots.draw scope
```

These read `runs/rq2/METRICS.csv` and `runs/rq3/METRICS.csv`. To render the small
archived reference results without running experiments, append `--reference`.
`--input` selects another CSV; `--config` uses the shared paths configuration.
PDF and PNG figures go to `paths.figures`. Rendering does not run experiments or
modify their results. The panels retain the paper's methods and quantities;
typography/layout are portable rather than manuscript-layout dependencies.
