# Frozen paper references

These small CSVs are reference results, not outputs regenerated on the current
machine. `reference_metrics.csv` covers the main point and ablations;
`reference_scope.csv` covers all 16 H/K settings; `reference_timing.csv` contains
the paper's warm-cache timings (Intel Core i9-12900, two numerical-library
threads). Reruns belong under `runs/` and never overwrite these files.

Local macOS/ARM validation reproduces the main table (TECAD 180/23, Local 155/17,
Concat 84/17) and all listed ablation TP/FP counts. It does not reproduce every
floating-point score exactly: Mean+Std AP differs by about 0.000112, and 2 of 16
scope configurations have different threshold counts (H30/K5: four fewer FP;
H90/K5: two more TP). Other scope AP differences are at most 0.000504. For a
checked scope, original and regenerated D/C arrays are bitwise equal, while P
log-count coordinates differ by at most 6.7e-16 across platforms. Such changes
can alter near-tied rankings/threshold comparisons. No score rounding or
threshold tolerance is added to force agreement with the reference.

The primary fit/load/inference roundtrip and single-target replay are tested.
Full XCW execution requires the external runtime and facts; the checked-in
adapter tests do not constitute a native XCW rerun. A-N has been checked on a
32-target, 88-candidate real-record sample; full 5,191-target A-N timing requires
the separate normalized-record package.
