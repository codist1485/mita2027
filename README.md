# MiTA 2027 — person–job fit controlled evaluation

**Paper:** Learned but Not Transferred: Objective Modifications and Evaluation Choices in Frozen-Encoder Person--Job Fit  
**Authors:** XX1, XX2, XX3 (placeholders match the submitted manuscript)  
**Venue year:** 2027  

One-line summary: with a frozen MiniLM encoder on a Tianchi recruitment corpus, reverse job-quality reweighting and an auxiliary RJDB triplet loss do not detectably raise held-out ROC-AUC; follow-up checks show the auxiliary task is learned but does not transfer, and stronger interventions can harm AUC.

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
```

Timing (paper pipeline on RTX A4000): one converged E1-style run ≈ **5 minutes**;  
E1 40 runs ≈ 3–4 h wall-clock on 2 GPUs; DR 20 runs similar; E2 adds ~40 runs. Full matrix is on the order of **1–2 GPU-days**.

Smoke test (no external data): `python scripts/smoke_test.py`

## Paper ↔ artifacts map

| Paper item | Script / source | Output |
|---|---|---|
| Table of E1 cells / effects | `scripts/06_analyze.py`, macros | `results/e1_run_inventory.csv`, `results/results_macros.tex` |
| E2 user/job tables | same | `results/e2_run_inventory.csv` |
| DR dose–response | DR runs + `07_make_figures.py` | `results/dr_run_inventory.csv`, `figures/fig_dose.pdf` |
| Fig. effects (factorial) | paper analysis | `results/results_macros.tex` → `figures/fig_effects.pdf` |
| Fig. protocol (B0 gaps) | paper analysis | `figures/fig_protocol.pdf` |
| Manipulation check | MC scoring + job-cluster bootstrap | `results/mc_accuracy.csv`, `\\MCT` in `results_macros_v2.tex` |
| Quality subgroups (SG) | SG analysis | SG macros in `results_macros_v2.tex` |
| All numeric effects (6 d.p.) | — | `results/effects_summary.json` |

## Pre-registration

- `prereg/PREREG_2026-10-26.md` — primary E4/E1/E2 plan  
- `prereg/PREREG_ADDENDUM_2026-10-26b.md` — MC / DR / SG addendum  

In the original research repository these were committed as:

- freeze / v1 manuscript: `9d344289370e684ce39621cdf2aa29c31de9237f`
- addendum + paper_v2 scaffold (before MC/DR/SG runs): `b04a5dc03e4637a60821c1b00f1e678821bab96c`

## License

Code: MIT (see `LICENSE`). Data: Tianchi and RJDB terms apply; not redistributed here.

## Citation

```bibtex
@inproceedings{mita2027pjf,
   title={Learned but Not Transferred: Objective Modifications and Evaluation Choices in Frozen-Encoder {Person--Job} Fit},
  author={Lei Bu and Hong Pan and Xiaoqiang Di\textsuperscript{*}},
  year={2027},
  booktitle={MiTA}
}
```
