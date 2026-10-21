# Data inputs

Payloads are excluded from Git. Place inputs here, link existing directories, or
change `paths` in a YAML file passed with `--config`. Relative paths are always
resolved from the repository root. `manifest.json` pins the dataset and the final
paper feature revision. This repository does not download or redistribute data.

```text
data/
  dataset-V5/                  # immutable independent dataset package
    DATASET_LOCK.json
    SPLIT_MANIFEST.csv
    data/
      dataset.json
      features/D14.npz
      candidates/pool.parquet
      ...
  paper-features/               # required final paper description revision
    candidate_execution.parquet
    CTX_K20.npz
  record-replay/                # optional: RQ4 mode A-N
    MANIFEST.json
    records.sqlite
  XChainWatcher/               # optional: native XCW baseline
    cross-chain-rules-validator/datalog/
```

The independent V5 package contains an earlier candidate-feature revision.
The final experiments use `data-revision-grid-v1` execution/context descriptions
in `paper-features/`; those two files are required alongside V5, not a new target
split. Do not replace missing descriptions with zeros. Public data-download and
licensing arrangements remain to be supplied before release.

The frozen split has 8,657 reference, 1,111 scale and 1,113 calibration normal
transactions, then 5,000 normal and 191 candidate-anomaly evaluation transactions.

