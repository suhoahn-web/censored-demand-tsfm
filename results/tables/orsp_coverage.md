**Empirical coverage at the nominal quantiles the replenishment policy orders from, on uncensored evaluation days.**

| Method | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | mean \|error\| |
|---|---|---|---|---|---|---|
| C2 LoRA r16 (lr 1e-3) | 0.483 | 0.598 | 0.707 | 0.814 | 0.915 | 0.0107 |
| C2 LoRA r16 + mask | 0.479 | 0.591 | 0.698 | 0.81 | 0.912 | 0.011 |
| Chronos-2 | 0.464 | 0.585 | 0.697 | 0.811 | 0.916 | 0.0163 |
| Chronos-2 + mask | 0.485 | 0.594 | 0.697 | 0.808 | 0.912 | 0.0087 |
| TiRex | 0.422 | 0.548 | 0.676 | 0.803 | 0.922 | 0.0359 |
| Moirai-2 | 0.421 | 0.539 | 0.665 | 0.788 | 0.908 | 0.0392 |
| TimesFM | 0.441 | 0.558 | 0.678 | 0.795 | 0.911 | 0.0277 |
| LightGBM | 0.489 | 0.584 | 0.685 | 0.779 | 0.876 | 0.0175 |
| ETS+EQ | 0.525 | 0.626 | 0.725 | 0.817 | 0.904 | 0.0193 |
| Croston+EQ | 0.526 | 0.631 | 0.728 | 0.818 | 0.903 | 0.0211 |
| SBA+EQ | 0.523 | 0.628 | 0.724 | 0.816 | 0.901 | 0.0184 |
| TSB+EQ | 0.527 | 0.632 | 0.729 | 0.817 | 0.903 | 0.0217 |

*Each cell is the share of scored observations at or below the forecast quantile; perfect calibration equals the nominal level. The last column is the mean absolute deviation from nominal over the five levels.*
