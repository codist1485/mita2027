# Pre-specified analysis plan — Addendum (v2)

**Date fixed:** 2026-10-26, before any analysis or run described here was started.
**Commit this file before running anything; the commit timestamp is the record.**

## 0. Context and honesty statement

This addendum is written **after** the E4/E1/E2 results of the original plan (commit `390069472aed0fa9a5f123225dc23a0650e85bce`) were known: no detectable effect of reweighting (W) or the triplet loss (T). The analyses below were chosen *because* of that result, to test explanations for it. They are therefore confirmatory only with respect to the hypotheses stated here, and the paper will say that they were specified after the primary results were seen.

All three analyses use the existing pair-overlap split (seed 42), the converged budget (max 100 epochs, patience 10, best-validation checkpoint), the accelerated pipeline that passed the equivalence check, and the same two-stage bootstrap (B = 10,000; resample seeds, then test users; RNG seed 20261026 unless stated).

## 1. MC — Manipulation check: did the auxiliary task get learned?

- **Question.** Does the triplet term make the match head rank RJDB matched resumes above unmatched ones?
- **Data.** The official RJDB **test** split, encoded with the same frozen MiniLM. The training split is never used for this check; the dev split is unused.
- **Models.** The E1 best checkpoints of all four cells, seeds 1001–1010 (40 checkpoints). No retraining unless checkpoints are missing (see the prompt for the fallback).
- **Metric.** Triplet accuracy = mean over test triples of 1[s(r⁺, j) > s(r⁻, j)]. Resume embeddings go in the user slot, exactly as in training.
- **Estimand.** The factorial T main effect on accuracy, with the same formula as for AUC. Two-stage bootstrap: resample seeds, then test **jobs** (triples sharing a job are a cluster).
- **Interpretation, fixed in advance.**
  - T-accuracy effect > 0 with the interval excluding 0, while the AUC effect is null → the auxiliary task is learned but does not transfer to Tianchi.
  - Interval includes 0 → the auxiliary task is not learned at λ = 0.05, and the null AUC result is uninformative about triplet supervision as such.

## 2. DR — Dose-response: do stronger interventions change AUC?

- **New cells** (pair-overlap split, seeds 1001–1005, 20 runs):

  | Cell | Weights | λ |
  |---|---|---|
  | W-κ2 | w = clip(1 + 2(0.5 − q), 0.1, 3.0) | 0 |
  | W-κ4 | w = clip(1 + 4(0.5 − q), 0.1, 3.0) | 0 |
  | T-λ0.5 | w ≡ 1 | 0.5 |
  | T-λ2 | w ≡ 1 | 2.0 |

  κ = 1 reproduces the existing +W cell (w = 1.5 − q). All other settings are as in E1: 64 triples per batch, margin 1.0, same q.
- **Reference.** The E1 Base, +W (κ = 1), and +T (λ = 0.05) runs with seeds 1001–1005. These use the same seeds and pipeline, so the comparisons are paired.
- **Estimands.** For each new cell c: Δ_c = Ā_c − Ā_Base on test AUC, with a two-stage bootstrap over seeds {1001–1005} and test users.
- **Diagnostics.** Kish ESS/N of the training weights for each κ; final-epoch mean λL_tri / mean L_sup for each λ; RJDB test triplet accuracy for each λ (as in MC).
- **Interpretation.** No equivalence margin. If all Δ_c intervals contain 0, we report that a 4× (weighting) and 40× (triplet) stronger intervention also yields no detectable change, with the interval bounds. Any interval excluding 0 is reported with its sign, including harm.
- **Censoring rule.** Same as the original plan: if any DR run reaches epoch 100, all 20 DR runs are repeated with max 200 epochs.

## 3. SG — Quality subgroups: does reweighting help the jobs it targets?

- **Data.** E1 test predictions (seeds 1001–1010, all four cells); no new training.
- **Subgroups.** Tertiles of the job-quality score q. The cut points are computed on **training jobs** (not test jobs) and fixed before any test metric is computed. Test pairs are assigned by their job's q.
- **Estimands.** Within each tertile, pooled AUC per cell and the factorial W main effect. Also W_low − W_high (the gap between the lowest and highest tertile).
- **Uncertainty.** Two-stage bootstrap (seeds, then test users), resampling within the full test set so that tertile sizes vary across replicates.
- **Interpretation.** The hypothesis implied by the reverse weighting is W_low > 0 and W_low − W_high > 0. No multiplicity correction; the three tertile effects are reported together and read as descriptive.

## 4. Reporting rules

Same as the original plan: every run is reported; no seed, cell, or grid value is added, dropped, or changed after seeing results; the test set is never used for any choice; deviations are listed with reasons.
