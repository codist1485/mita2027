# Release scan notes

## Sensitive-keyword scan

| Pattern | Hits | Disposition |
|---|---|---|
| `token` | `src/models/structured_mlp.py`, `src/phase1/quality_train_only.py` | **Benign** — NLP tokenizer / feature field names (`n_tokens`, token counts), not API secrets |
| `ssh` / `paramiko` / `password` / `api_key` / `BEGIN ` / `C:\Users` / `/home/` / emails / IPv4 | none | — |

## Data-leak scan

- Files > 1 MB: **none**
- CSV/JSON containing `user_id`+`jd_no` data rows: **none** (inventories are aggregate run metrics only)

## Action taken

No redaction required beyond the builder’s path scrubbing when copying sources.
