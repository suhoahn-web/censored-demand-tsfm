**Table C5.** Signed error of the median forecast (WPE, as in the main text) against the signed error of the mean of the predictive distribution approximated from the quantile grid (trapezoid over the reported levels, outer tails held at the extreme quantiles); LightGBM also with its Tweedie point forecast. FreshRetailNet-LT, uncensored days; adapted Chronos-2 configurations are seed 0, whereas Tables C1 and C3 average three seeds. Point-only methods have no grid mean.

| Model | Signed median error (WPE, %) | Signed mean error (grid mean, %) | Median − mean component (pp) | Tweedie mean error (%) |
|---|---|---|---|---|
| TTM + mask | -24.67 | – | – | – |
| SBA | -13.16 | – | – | – |
| TiRex | -9.96 | -8.05 | -1.91 | – |
| Moirai-2 | -9.86 | -8.68 | -1.18 | – |
| TSB | -8.94 | – | – | – |
| Croston | -8.59 | – | – | – |
| TimesFM | -8.23 | -6.60 | -1.62 | – |
| Moirai-2 + mask | -7.45 | -5.91 | -1.54 | – |
| Chronos-2 | -6.17 | -5.18 | -0.99 | – |
| C2 full-FT | -5.93 | -5.70 | -0.23 | – |
| C2 LoRA r8 | -5.64 | -5.24 | -0.40 | – |
| C2 LoRA r32 | -5.62 | -5.24 | -0.38 | – |
| C2 LoRA r16 | -5.60 | -5.24 | -0.35 | – |
| LightGBM | -5.36 | -6.65 | +1.28 | -6.42 |
| C2 LoRA r16 (lr 1e-5) | -5.29 | -4.15 | -1.14 | – |
| TiRex + mask | -4.98 | -1.36 | -3.61 | – |
| sNaive | -4.96 | – | – | – |
| C2 LoRA r16 (lr 1e-3) | -4.60 | -4.61 | +0.02 | – |
| ETS | -4.26 | – | – | – |
| TTM+EQ | -2.84 | -3.19 | +0.35 | – |
| C2 LoRA r16 + mask | -2.67 | +0.05 | -2.72 | – |
| SBA+EQ | -2.54 | -2.32 | -0.22 | – |
| Croston+EQ | -2.29 | -2.49 | +0.19 | – |
| TSB+EQ | -2.28 | -2.36 | +0.08 | – |
| Chronos-2 + mask | -1.92 | +1.17 | -3.10 | – |
| ETS+EQ | -1.33 | -2.46 | +1.13 | – |
| sNaive+EQ | -1.21 | -0.21 | -1.00 | – |
| TimesFM + mask | -0.90 | +0.30 | -1.20 | – |
| TTM | +0.47 | – | – | – |