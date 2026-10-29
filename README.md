# MiTA 2027 — person–job fit controlled evaluation

**Paper:** Learned but Not Transferred: Evaluation Choices Outweigh Objective Modifications in Frozen-Encoder Person--Job Fit  
**Authors:** Lei Bu, Hong Pan, Xiaoqiang Di  
**Venue year:** 2027  

## Abstract

Person–job fit (PJF) models are often compared on a random split of interaction pairs with pooled AUC. We show that, for a frozen-encoder matcher on a Tianchi recruitment corpus, this protocol and the training budget decide the conclusion more than the training objective does. Training to convergence instead of for 10 epochs raises AUC by +0.0111. A text-free label-frequency prior beats the text scorer by +0.0215 at 10 epochs, is not separable from it at convergence, and loses by −0.0211 when test jobs are unseen in training. Against these shifts, two inexpensive objective modifications motivated by recent work have no detectable effect on any split in a 2×2 factorial design: an auxiliary triplet loss on an LLM-generated resume–job benchmark (RJDB), and reweighting toward short, low-quality postings. On the pair-overlap split, gains above about 0.003 AUC are not supported. The auxiliary task is learned—RJDB triplet accuracy rises from 0.61 to 0.89—but does not transfer, and stronger versions of either modification lower AUC. Comparisons of training objectives in PJF should therefore report convergence, entity overlap, and whether auxiliary tasks are learned.

## Main findings

**C1 — Evaluation choices dominate.** Training budget and entity overlap move AUC far more than either objective at default strength (budget gain +0.0111); B0's lead tracks job/user training history (addendum e).

**C2 — LLM-generated supervision is learned but not transferred.** RJDB triplet accuracy rises from 0.61 to 0.89 while Tianchi AUC does not rise; larger λ lowers AUC.

**C3 — Quality reweighting does not help at any strength.** Default weights are nearly uniform; stronger composite or length-only reweighting lowers AUC.

> This repository ships **code and aggregate results only**. It does **not** redistribute Tianchi or RJDB data, embeddings, checkpoints, or pair-level predictions.

## Data access

- **Tianchi:** Second Alibaba Big Data Intelligent Cloud Programming Competition — Zhaopin Intelligent Person–Job Matching (Preliminary-Round Data).  
  https://tianchi.aliyun.com/dataset/44080  
  (BibTeX key `tianchi_pjf` in the paper.)
- **RJDB:** https://github.com/megagonlabs/rjdb  
- See `scripts/00_download_instructions.md` for the expected local layout.

## Environment

- Python 3.10+ recommended  
- Paper GPU: NVIDIA RTX A4000  
- Install: `python -m venv .venv && source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows`  
  `pip install -r requirements.txt`

## Reproduce (after obtaining data)

```bash
python scripts/01_clean_tianchi.py
python scripts/02_make_splits.py --protocol overlap --seed 42
python scripts/02_make_splits.py --protocol user --seed 2027
python scripts/02_make_splits.py --protocol job --seed 2028
python scripts/03_encode_minilm.py --device cuda
bash scripts/05_run_all.sh          # or run scripts/04_train.py per cell
python scripts/06_analyze.py
python scripts/07_make_figures.py   # writes PDF vectors under figures/
# optional post-hoc history analysis (addendum e; no training):
# python scripts/v4e_history_strata.py
```

Timing (paper pipeline on RTX A4000): one converged E1-style run ≈ **5 minutes**;  
E1 40 runs ≈ 3–4 h wall-clock on 2 GPUs; DR 20 runs similar; E2 adds ~40 runs. Full matrix is on the order of **1–2 GPU-days**.

Smoke test (no external data): `python scripts/smoke_test.py`

## Paper ↔ artifacts map

| Paper item | Script / source | Output |
|---|---|---|
| Table I: data & splits | `scripts/02_make_splits.py` | split CSVs / `results/e2` stats |
| Table II: findings summary | paper + macros | `results/results_macros*.tex` |
| Table III: history strata (B0−Base) | `scripts/v4e_history_strata.py` | `results/results_macros_v4e.tex`, `results/v4e_cutpoints.json` |
| Table IV: AUC | `scripts/06_analyze.py` | `results/e1_run_inventory.csv`, `results/e2_run_inventory.csv` |
| Table V: factor effects | `scripts/06_analyze.py` | `results/effects_summary.json` |
| Table VI: dose–response | DR / length-only + macros | `results/dr_run_inventory.csv`, `results/results_macros_v3.tex`, `results/results_macros_v3d.tex` |
| Table VII: checklist | paper | — |
| Fig. 1: validation curves | `scripts/make_fig_seeds_curves.py` | `figures/fig_curves.pdf` |
| Fig. 2: B0 − Base (protocol) | paper analysis | `figures/fig_protocol.pdf` |
| Fig. 3: per-seed AUC | `scripts/make_fig_seeds_curves.py` | `figures/fig_seeds.pdf` |
| Fig. 4: dose–response | `scripts/07_make_figures.py` | `figures/fig_dose.pdf` |
| Manipulation check | MC scoring + job-cluster bootstrap | `results/mc_accuracy.csv`, `\\MCT` in `results_macros_v2.tex` |
| All numeric effects (6 d.p.) | — | `results/effects_summary.json` (incl. `v4e`) |

## Pre-registration

- `prereg/PREREG_2026-10-26.md` — primary E4/E1/E2 plan  
- `prereg/PREREG_ADDENDUM_2026-10-26b.md` — MC / DR / SG addendum  
- `prereg/PREREG_ADDENDUM_2026-10-27c.md` — short-budget 10-seed table, AP factorial effects, DR Bonferroni / slopes (post hoc re-analysis)  
- `prereg/PREREG_ADDENDUM_2026-10-27d.md` — length-only reverse reweighting dose grid (before L-κ runs)  
- `prereg/PREREG_ADDENDUM_2026-10-29e.md` — B0 vs Base by training history (before history-strata AUCs)

In the original research repository these were committed as:

- freeze / v1 manuscript: `9d344289370e684ce39621cdf2aa29c31de9237f`
- addendum + paper_v2 scaffold (before MC/DR/SG runs): `b04a5dc03e4637a60821c1b00f1e678821bab96c`
- addendum c (before v3 re-analyses): `1bf856140d9be6f257bbff1152b0cf729ee2a2db`
- addendum d (before length-only runs): `d9dae5cbdd950f5b6f18c260ae03d7943c40e338`
- addendum e (before history strata): `bc49ebbaaf5091d796f7744894cf195fa9334885`

## License

Code: MIT (see `LICENSE`). Data: Tianchi and RJDB terms apply; not redistributed here.

## Citation

```bibtex
@inproceedings{mita2027pjf,
  title={Learned but Not Transferred: Evaluation Choices Outweigh Objective Modifications in Frozen-Encoder Person--Job Fit},
  author={Lei Bu and Hong Pan and Xiaoqiang Di},
  year={2027},
  booktitle={MiTA}
}
```
