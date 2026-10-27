**Sensitivity of the inventory comparison to the assumed evaluation demand path.**

| q | b | obs. vs uplift | obs. vs clean | cheapest (observed) | cheapest (uplift) |
|---|---|---|---|---|---|
| 0.5 | 0.3 | 0.72 | 0.51 | LightGBM | TSB+EQ |
| 0.5 | 1 | 0.83 | 0.68 | ETS+EQ | TSB+EQ |
| 0.5 | 3 | 0.89 | 0.75 | ETS+EQ | TSB+EQ |
| 0.7 | 0.3 | 0.48 | 0.34 | LightGBM | TSB+EQ |
| 0.7 | 1 | 0.48 | 0.24 | C2 LoRA r16 (lr 1e-3) | C2 LoRA r16 + mask |
| 0.7 | 3 | 0.52 | 0.34 | C2 LoRA r16 (lr 1e-3) | C2 LoRA r16 + mask |
| 0.9 | 0.3 | 0.77 | 0.70 | LightGBM | C2 LoRA r16 (lr 1e-3) |
| 0.9 | 1 | 0.37 | 0.76 | LightGBM | C2 LoRA r16 (lr 1e-5) |
| 0.9 | 3 | 0.29 | 0.61 | C2 LoRA r16 (lr 1e-3) | TiRex + mask |

*Entries are Spearman rank correlations of cost across the 28 methods between demand paths, and the cheapest method under each path. The observed path uses recorded sales, the uplift path applies the hour-proportional correction of Section 3.3, and the clean path restricts to the 340 series with no stockout in the evaluation week.*
