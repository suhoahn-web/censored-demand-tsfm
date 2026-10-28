**Table C9 (GPU part).** Zero-shot inference on the LT evaluation series, one A100 80GB, batch size 256: wall-clock and peak GPU memory (torch.cuda.max_memory_allocated / reserved). Measured 2026-10-27.

| Model | Series measured | Wall-clock (s) | Seconds per 10k series | Peak GPU memory allocated (GB) | Peak reserved (GB) |
|---|---|---|---|---|---|
| Chronos-2 | 10,000 | 11.5 | 11.5 | 0.86 | 0.99 |
| Moirai-2.0-R-small | 10,000 | 14.6 | 14.6 | 0.21 | 0.31 |
| TimesFM-2.5 | 1,000 | 169.2 | 1692.2 | 0.88 | 0.95 |
| TiRex | 10,000 | 26.4 | 26.4 | 0.55 | 0.79 |
| TTM-R2 | 10,000 | 5.9 | 5.9 | 0.03 | 0.05 |