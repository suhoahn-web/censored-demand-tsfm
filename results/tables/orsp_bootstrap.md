**Inventory cost with store-level cluster bootstrap intervals in the shortage-dominated cell ($b = 3.0$, $q = 0.9$); ten cheapest methods.**

| Method | cost / unit demand | difference vs ETS+EQ | achieved fill rate |
|---|---|---|---|
| C2 LoRA r16 (lr 1e-3) | 0.1094 [0.1053, 0.1139] | -0.0335 [-0.0376, -0.0295] | 0.984 |
| C2 LoRA r16 (lr 1e-5) | 0.1106 [0.1065, 0.1152] | -0.0323 [-0.0363, -0.0282] | 0.985 |
| C2 full-FT | 0.1125 [0.1085, 0.1169] | -0.0305 [-0.0346, -0.0264] | 0.983 |
| TiRex | 0.1129 [0.1088, 0.1174] | -0.0301 [-0.0338, -0.0261] | 0.984 |
| C2 LoRA r16 | 0.1131 [0.1091, 0.1175] | -0.0299 [-0.0339, -0.0256] | 0.983 |
| C2 LoRA r32 | 0.1132 [0.1091, 0.1175] | -0.0298 [-0.0338, -0.0255] | 0.983 |
| C2 LoRA r8 | 0.1132 [0.1091, 0.1176] | -0.0298 [-0.0338, -0.0255] | 0.983 |
| Chronos-2 | 0.1156 [0.1112, 0.1203] | -0.0274 [-0.0314, -0.0232] | 0.983 |
| Moirai-2 | 0.1176 [0.1131, 0.1221] | -0.0254 [-0.0288, -0.0218] | 0.980 |
| TimesFM | 0.1181 [0.1133, 0.1232] | -0.0249 [-0.0278, -0.0218] | 0.980 |

*Brackets give 95% percentile intervals from 2,000 store-level cluster bootstrap replicates. A difference whose interval excludes zero is distinguishable from the reference; the achieved fill rate differs across methods even though all order at the same quantile.*
