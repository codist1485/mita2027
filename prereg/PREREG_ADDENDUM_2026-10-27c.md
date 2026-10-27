# Analysis addendum c — 2026-10-27

Written after the v2 results (E1, E2, DR, SG, MC) were known, in response to reviewer feedback.
It adds only re-analyses of existing predictions; no new training.

1. Short-budget 10-seed means (AUC, AP) for Base, +W, +W+T, already used in the text, now tabulated.
2. AP factorial effects with the same two-stage bootstrap as AUC (B=10,000; seed 20261026).
3. Dose–response robustness: Bonferroni-adjusted intervals (98.75% for the 4 new dose contrasts;
   98.33% for the 3 quality tertiles) and least-squares trend slopes of ΔAUC on log2(κ) and log10(λ),
   computed within each bootstrap replicate with shared seed and user resamples.

All are post hoc and will be labelled as such in the paper. No contrast, seed, or grid value is added
or removed.
