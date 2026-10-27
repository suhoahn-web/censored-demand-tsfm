**Table B7.** Controlled censoring of the evaluation target on fully observed series: inventory cost per unit of demand when the same forecasts are scored against true demand and against synthetically censored sales.

*LT (340 series), b = 3.0, q = 0.9, mean of five censoring draws.*

| Model | Cost vs true demand | Cost vs censored target | Inflation % | Rank (true) | Rank (censored) |
|---|---|---|---|---|---|
| Chronos-2 + mask | 0.0774 | 0.1601 | 107 | 1 | 11 |
| C2 LoRA r16 + mask | 0.0783 | 0.1441 | 84 | 2 | 1 |
| C2 LoRA r16 (lr 1e-5) | 0.0800 | 0.1589 | 98 | 3 | 10 |
| C2 LoRA r8 | 0.0825 | 0.1516 | 84 | 4 | 5 |
| C2 LoRA r16 | 0.0826 | 0.1516 | 83 | 5 | 3 |
| C2 full-FT | 0.0827 | 0.1516 | 83 | 6 | 4 |
| C2 LoRA r32 | 0.0830 | 0.1519 | 83 | 7 | 6 |
| TiRex + mask | 0.0831 | 0.1698 | 104 | 8 | 16 |
| Moirai-2 | 0.0864 | 0.1537 | 78 | 9 | 8 |
| Chronos-2 | 0.0870 | 0.1608 | 85 | 10 | 12 |
| TiRex | 0.0870 | 0.1638 | 88 | 11 | 13 |
| C2 LoRA r16 (lr 1e-3) | 0.0875 | 0.1527 | 74 | 12 | 7 |
| TimesFM | 0.0902 | 0.1568 | 74 | 13 | 9 |
| Moirai-2 + mask | 0.0924 | 0.1678 | 82 | 14 | 15 |
| LightGBM | 0.0980 | 0.1455 | 48 | 15 | 2 |
| TimesFM + mask | 0.1053 | 0.1651 | 57 | 16 | 14 |
| ETS+EQ | 0.1206 | 0.1781 | 48 | 17 | 17 |
| TSB+EQ | 0.1670 | 0.2125 | 27 | 18 | 18 |
| Croston+EQ | 0.1699 | 0.2148 | 26 | 19 | 19 |
| SBA+EQ | 0.1748 | 0.2192 | 25 | 20 | 20 |
| sNaive+EQ | 0.2054 | 0.2696 | 31 | 21 | 21 |

*50K (5,874 series), b = 3.0, q = 0.9, mean of five censoring draws.*

| Model | Cost vs true demand | Cost vs censored target | Inflation % | Rank (true) | Rank (censored) |
|---|---|---|---|---|---|
| lightgbm_frn50k_raw | 0.1467 | 0.2183 | 49 | 1 | 1 |
| etseq_frn50k_raw | 0.1549 | 0.2621 | 69 | 2 | 8 |
| ft_c2_50k_lora_r16_s0 | 0.1581 | 0.2660 | 68 | 3 | 10 |
| crostoneq_frn50k_raw | 0.1601 | 0.2562 | 60 | 4 | 3 |
| tsbeq_frn50k_raw | 0.1605 | 0.2554 | 59 | 5 | 2 |
| sbaeq_frn50k_raw | 0.1621 | 0.2567 | 58 | 6 | 4 |
| tirex_50k_raw | 0.1649 | 0.2588 | 57 | 7 | 6 |
| timesfm_50k_raw | 0.1746 | 0.2581 | 48 | 8 | 5 |
| moirai2_50k_raw | 0.1760 | 0.2589 | 47 | 9 | 7 |
| chronos2_50k_raw | 0.1768 | 0.2647 | 50 | 10 | 9 |
| tirex_frn50k_mask | 0.1772 | 0.2739 | 55 | 11 | 12 |
| chronos2_frn50k_mask | 0.1813 | 0.2675 | 48 | 12 | 11 |
| moirai2_frn50k_mask | 0.2028 | 0.2773 | 37 | 13 | 13 |
| snaiveeq_frn50k_raw | 0.2093 | 0.3347 | 60 | 14 | 15 |
| timesfm_50k_mask | 0.2151 | 0.2788 | 30 | 15 | 14 |