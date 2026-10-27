# SYNC_REPORT — release → release_sync (v3)

**UTC:** 2026-10-27  
**Note:** `CURSOR_PROMPT_repo_sync.md` was **not found** in the project tree; section 3 was executed as a paper_v3 package update (artifacts listed below), then robocopy + commit in `release_sync/mita2027`.

## 1. Prereg check

| File | In `release/mita2027/prereg/` before | Action |
|---|---|---|
| `PREREG_ADDENDUM_2026-10-27c.md` | absent | copied from project root |
| `PREREG_ADDENDUM_2026-10-27d.md` | absent | copied from project root |

Original commits (research repo): c=`1bf8561…`, d=`d9dae5c…`.

## 2. Release package updates (v3)

- Synced (sanitized) `src/phase1/{quality_train_only,train_entry,losses}.py`, `src/models/quality_aware_dual_encoder.py` (length-only `--quality-score-column`).
- Added `results/results_macros_v3.tex`, `results/results_macros_v3d.tex`; refreshed `results_macros_v2.tex` from paper_v3.
- Added `results/len_run_inventory.csv` (15 L-κ runs).
- Extended `results/effects_summary.json` with `v3` block (AP effects, DR Bonferroni/slopes, length-only, post-hoc MacroBudgetGap).
- README: four prereg files + v3 artifact map rows.
- Scripts: `04_train.py` cells `lk1/lk2/lk4`; `05_run_all.sh` length-only loop; `07_make_figures.py` composite + length-only left panel.

## 3. Robocopy

```text
robocopy D:\work\mita\release\mita2027 D:\work\mita\release_sync\mita2027 /E /XD .git .venv __pycache__
```

Copied 15 files (robocopy exit 1 = success with copies). `.venv` / `__pycache__` excluded (gitignored).

## 4. Git (release_sync/mita2027)

Staged all package changes; committed; **did not push**.

## git diff --cached --stat
 .gitignore                            |  11 ++++
 README.md                             |  11 +++-
 SYNC_REPORT.md                        |  36 ++++++++++++
 prereg/PREREG_ADDENDUM_2026-10-27c.md |  13 +++++
 prereg/PREREG_ADDENDUM_2026-10-27d.md |  55 ++++++++++++++++++
 results/effects_summary.json          | 101 ++++++++++++++++++++++++++++++++++
 results/len_run_inventory.csv         |  16 ++++++
 results/results_macros_v3.tex         |  35 ++++++++++++
 results/results_macros_v3d.tex        |  21 +++++++
 scripts/04_train.py                   |  11 +++-
 scripts/05_run_all.sh                 |   7 +++
 scripts/07_make_figures.py            |  49 +++++++++++++++--
 src/phase1/quality_train_only.py      |  15 ++++-
 src/phase1/train_entry.py             |  10 +++-
 14 files changed, 381 insertions(+), 10 deletions(-)
