# Data download instructions

This repository does **not** redistribute Tianchi or RJDB data.

## Tianchi (person–job matching)

- Official name: Second Alibaba Big Data Intelligent Cloud Programming Competition:
  Zhaopin Intelligent Person–Job Matching (Preliminary-Round Data)
- URL: https://tianchi.aliyun.com/dataset/44080
- Place extracted tables under `data/raw/tianchi/` (see `01_clean_tianchi.py` for expected filenames).

## RJDB

- GitHub: https://github.com/megagonlabs/rjdb
- Place JSONL under `data/raw/rjdb/` (train/dev/test as released).
- Do not commit embeddings or JSONL into this repository.

## Expected layout after download

```
data/
  raw/tianchi/   # provider tables
  raw/rjdb/      # JSONL from Megagon Labs
  processed/     # written by 01_clean_tianchi.py
  embeddings/    # written by 03_encode_minilm.py
  splits/        # written by 02_make_splits.py
```
