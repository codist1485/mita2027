# Lambda selection policy (frozen)

Status: protocol frozen; search not executed in Phase 1B+.

## Allowed metric

- Only `val_auc` from the frozen Phase 1 validation split.
- Forbidden: any `rank_job` / `rank_resume` metric, any `internal_test` metric.

## Candidates

`{0.0, 0.01, 0.05, 0.1}`

## Sharing rule

- A4 and A5 do **not** share a single lambda.
- Select lambda for A4 and A5 independently.

## Procedure

1. For each experiment in `{A4, A5}` and each candidate lambda:
   - run seeds `{42, 123, 456}` with identical budget (`max_epochs=10`, patience=3, batch=512, lr=1e-3).
2. Score = mean best-epoch `val_auc` across the three seeds.
3. Pick the candidate with the highest mean. Ties → smaller lambda.
4. Write the winners into `configs/phase1/matrix.json` and `results/phase1/frozen_config.json`.

## Freeze and test access

- After winners are written, treat A0–A5/S1/S2 as frozen.
- First `internal_test` evaluation is allowed only after that freeze file exists.
- Official rank remains optional, post-hoc, and never used for selection.
