**Table C7.** MASE with every series weighted equally (main text) and weighted by its number of uncensored evaluation days (1–7). No series has a zero MASE scale or zero scored days; adapted Chronos-2 configurations are seed 0.

| Model | MASE (equal weights) | MASE (weighted by scored days) | Rank (equal) | Rank (weighted) |
|---|---|---|---|---|
| LightGBM | 0.841 | 0.842 | 1 | 1 |
| C2 full-FT | 0.851 | 0.847 | 2 | 2 |
| C2 LoRA r16 (lr 1e-3) | 0.854 | 0.849 | 3 | 3 |
| C2 LoRA r16 | 0.856 | 0.851 | 4 | 4 |
| C2 LoRA r32 | 0.857 | 0.851 | 5 | 5 |
| C2 LoRA r8 | 0.857 | 0.851 | 6 | 6 |
| C2 LoRA r16 (lr 1e-5) | 0.866 | 0.860 | 7 | 7 |
| Chronos-2 | 0.878 | 0.871 | 8 | 8 |
| TimesFM | 0.884 | 0.874 | 9 | 9 |
| ETS | 0.885 | 0.879 | 10 | 10 |
| ETS+EQ | 0.887 | 0.882 | 11 | 15 |
| Croston+EQ | 0.890 | 0.879 | 12 | 11 |
| SBA+EQ | 0.890 | 0.880 | 13 | 12 |
| TSB+EQ | 0.891 | 0.880 | 14 | 13 |
| C2 LoRA r16 + mask | 0.898 | 0.881 | 15 | 14 |
| Moirai-2 | 0.901 | 0.889 | 16 | 16 |
| TiRex | 0.908 | 0.893 | 17 | 17 |
| TTM+EQ | 0.940 | 0.924 | 18 | 20 |
| Croston | 0.947 | 0.922 | 19 | 18 |
| TSB | 0.947 | 0.923 | 20 | 19 |
| TiRex + mask | 0.964 | 0.938 | 21 | 21 |
| SBA | 0.967 | 0.941 | 22 | 23 |
| Chronos-2 + mask | 0.973 | 0.940 | 23 | 22 |
| TTM | 1.019 | 0.990 | 24 | 24 |
| TimesFM + mask | 1.040 | 1.000 | 25 | 25 |
| Moirai-2 + mask | 1.054 | 1.021 | 26 | 26 |
| sNaive | 1.125 | 1.129 | 27 | 27 |
| sNaive+EQ | 1.143 | 1.146 | 28 | 28 |
| TTM + mask | 1.451 | 1.365 | 29 | 29 |