#!/usr/bin/env bash
# Reproduce paper runs (E4 short budget, E1 converged, E2 entity-disjoint, DR dose grid).
# Each E1-style run ≈ 5 minutes on NVIDIA RTX A4000 with the accelerated pipeline.
set -euo pipefail
PY="${PYTHON:-python}"
OUT="${OUTPUT_ROOT:-results/runs}"

# E1: 4 cells × seeds 1001–1010 (40 runs), max 100 ep, patience 10
for cell in base w t wt; do
  for s in $(seq 1001 1010); do
    $PY scripts/04_train.py --cell "$cell" --seed "$s" --max-epochs 100 --patience 10 --output-root "$OUT/e1"
  done
done

# E4: same cells/seeds, max 10 ep (re-analyse short budget)
for cell in base w t wt; do
  for s in $(seq 1001 1010); do
    $PY scripts/04_train.py --cell "$cell" --seed "$s" --max-epochs 10 --patience 10 --output-root "$OUT/e4"
  done
done

# DR: κ∈{2,4}, λ∈{0.5,2}, seeds 1001–1005
for cell in wk2 wk4 tl05 tl2; do
  for s in $(seq 1001 1005); do
    $PY scripts/04_train.py --cell "$cell" --seed "$s" --max-epochs 100 --patience 10 --output-root "$OUT/dr"
  done
done

echo "E2 user/job-disjoint runs: use --split-protocol flags documented in README (20+20 runs)."
echo "Then: $PY scripts/06_analyze.py && $PY scripts/07_make_figures.py"
