# Pre-specified analysis plan — Addendum d (length-only reweighting)

**Date fixed:** 2026-10-27, before any run described here was started.
**Commit this file before launching any run; the commit timestamp is the record.**

## 0. Context and honesty statement

This addendum is written after the v2 results were known. In particular, composite-quality reweighting had no detectable effect at κ = 1 and lowered AUC at κ = 4. A reviewer noted that the composite quality score q_j (0.4·q_len + 0.4·q_struct + 0.2·q_kw) is an unvalidated heuristic, so a null result may be specific to that formula. Recent industrial work identifies short job descriptions as a specific weak spot.

This addendum tests whether the conclusion changes when weights depend **only on description length**, the simplest and most transparent quality proxy. It is post hoc relative to the v2 results and will be labelled as such.

## 1. Design

**Weights.**

    w_j = clip(1 + κ·(0.5 − q_j^len), 0.1, 3.0)

Here q_j^len is the existing length score: description length rescaled between the 5th and 95th percentiles of training-job lengths (248 and 845 characters), clipped to [0, 1], and fit on training jobs only. Jobs without text receive q^len = 0.5. This is the same functional form as the composite dose–response cells, with q_j replaced by q_j^len.

**New cells** (pair-overlap split, seed 42; 15 runs in total):

| Cell | Weights | λ | Seeds |
|---|---|---|---|
| L-κ1 | κ = 1 | 0 | 1001–1005 |
| L-κ2 | κ = 2 | 0 | 1001–1005 |
| L-κ4 | κ = 4 | 0 | 1001–1005 |

**Fixed settings.** Everything else is exactly as in E1 and DR:
- Accelerated pipeline that passed the equivalence check
- Adam, learning rate 1e-3, batch 512, positive-class weight unchanged
- Max 100 epochs, early-stopping patience 10 on validation ROC-AUC, best-validation checkpoint

**Reference.** E1 Base runs, seeds 1001–1005 (same seeds and pipeline, so comparisons are paired). The composite-q curve (+W, DR W-κ2, DR W-κ4) is reported alongside for comparison but is not re-run.

**Censoring rule.** If any of the 15 runs reaches epoch 100 without early stopping, all 15 are repeated with max 200 epochs.

## 2. Estimands and uncertainty

- **Primary quantity for this addendum.** Δ_κ = Ā(L-κ) − Ā(Base) on pooled test ROC-AUC, for κ ∈ {1, 2, 4}.
- **Interval.** Two-stage percentile bootstrap: resample seeds, then test users; B = 10,000; RNG seed 20261026.
- **Multiplicity.** Bonferroni-adjusted 98.33% intervals over the three contrasts.
- **Trend.** Least-squares slope of Δ on log₂κ ∈ {0, 1, 2}, computed within each bootstrap replicate with shared resamples.
- **Diagnostic.** Kish ESS/N of the training weights for each κ.

## 3. Interpretation, fixed in advance

- **All three intervals contain 0, or harm appears only at κ = 4.** The v2 conclusion does not depend on the composite formula: length-based reweighting behaves the same way.
- **An interval excludes 0 in the positive direction.** Length-only weighting helps where composite weighting did not. The paper will then report that the composite heuristic, not reweighting toward short postings as such, was ineffective, and will narrow its conclusion accordingly.
- **Any other pattern** is reported as observed, with its sign.

No equivalence margin is declared.

## 4. Reporting rules

Every run is reported. No seed, κ value, or setting is added, dropped, or changed after seeing results. The test set is not used for any choice. Deviations are listed with reasons.
