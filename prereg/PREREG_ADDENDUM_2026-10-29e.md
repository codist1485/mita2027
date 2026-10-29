# Analysis addendum e — 2026-10-29 (B0 advantage by training history)

Written after all earlier results were known, and before the computation below. This is a re-analysis of existing predictions; no training is involved.

## Question

On the pair-overlap split, does the advantage of the text-free prior B0 over converged Base grow with the amount of training history of the test job or the test user?

Prediction, if B0's strength comes from entity recurrence: the gap B0 − Base is larger in the highest-history third than in the lowest.

## Data

- B0 test predictions on the pair-overlap test set (seed 42 split).
- E1 Base test predictions, seeds 1001–1010.

## History

- For each test pair, the job history is the number of training pairs of that job, and the user history is the number of training pairs of that user.
- Cut points are the 1/3 and 2/3 quantiles of the history values over **test pairs**.
- Ties at a cut point go to the lower third.
- Cut points are computed and written to file before any AUC is computed.
- Jobs and users are stratified separately, giving six strata.

## Estimands

Within each stratum:
- pooled AUC of B0
- seed-mean pooled AUC of Base
- Gap = AUC(B0) − AUC(Base)

Trend = Gap(high third) − Gap(low third), computed separately for jobs and for users.

## Uncertainty

Two-stage bootstrap with B = 10,000 and RNG seed 20261029:
1. Resample Base seeds. B0 is deterministic and is not resampled at this stage.
2. Resample test users with all their pairs.
3. Recompute strata AUCs on the resampled pairs, using the cut points fixed above. Strata are not re-cut in each replicate.

Report 95% percentile intervals. If a replicate leaves a stratum without a positive or a negative pair, skip that replicate and report the number skipped.

## Reporting

- All six strata and both trends are reported.
- No multiplicity adjustment.
- The analysis is labelled post hoc ("F") in the paper.
