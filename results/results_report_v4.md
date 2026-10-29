# Results report v4

## Addendum e — B0 vs Base by training history

- Prereg commit: `bc49ebbaaf5091d796f7744894cf195fa9334885`
- Cutpoints file: `D:/work/mita/results/v4e_cutpoints.json`
- Cutpoints written_utc: `2026-10-29T02:59:14.974241+00:00`
- Job cutpoints (test-pair quantiles 1/3, 2/3): q33=3.0, q66=11.0
- User cutpoints (test-pair quantiles 1/3, 2/3): q33=30.0, q66=54.0
- Bootstrap: B=10000, RNG seed=20261029
- Skipped replicates (stratum missing a class): **0**
- Successful replicates: **10000**

### Point estimates (6 d.p.)

#### Job history
- Low: n=5862 range=0--3 AUC_B0=0.615812 AUC_Base=0.652393 Gap=-0.036581 95%CI=[-0.056289, -0.016800]
- Mid: n=4456 range=4--11 AUC_B0=0.676469 AUC_Base=0.661994 Gap=+0.014475 95%CI=[-0.010152, +0.038706]
- High: n=5089 range=12--210 AUC_B0=0.706355 AUC_Base=0.642759 Gap=+0.063596 95%CI=[+0.041882, +0.085081]
- Trend (High−Low Gap): +0.100178 95%CI=[+0.070450, +0.130154]

#### User history
- Low: n=5297 range=0--30 AUC_B0=0.623079 AUC_Base=0.633331 Gap=-0.010251 95%CI=[-0.031904, +0.011903]
- Mid: n=5115 range=31--54 AUC_B0=0.669703 AUC_Base=0.646782 Gap=+0.022921 95%CI=[+0.001983, +0.043860]
- High: n=4995 range=55--250 AUC_B0=0.707557 AUC_Base=0.682503 Gap=+0.025054 95%CI=[+0.003114, +0.047080]
- Trend (High−Low Gap): +0.035305 95%CI=[+0.003379, +0.066159]

