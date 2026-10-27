**Forecast accuracy on FreshRetailNet-50K (official 7-day evaluation, days without stockouts only). WAPE and WPE in \%; WPE $<0$ indicates under-forecasting.**

| Model            | Family            |   WAPE |   MASE |    WQL |   CRPS |    WPE |
|:-----------------|:------------------|-------:|-------:|-------:|-------:|-------:|
| TFT [rec]        | Local DL          |  28.42 |  0.838 | 0.2842 | 0.3516 |  -0.04 |
| LightGBM         | Tree/ML           |  28.97 |  0.823 | 0.2276 | 0.2815 |  -5.8  |
| TFT [rec]        | Local DL          |  29.13 |  0.862 | 0.2913 | 0.3604 |   4.36 |
| TFT [rec]        | Local DL          |  30.41 |  0.904 | 0.3041 | 0.3762 |   8.94 |
| DLinear [rec]    | Local DL          |  30.55 |  0.889 | 0.3055 | 0.3779 |   6.2  |
| TFT              | Local DL          |  30.78 |  0.843 | 0.3078 | 0.3808 |  -6.96 |
| C2 LoRA r16      | TSFM (fine-tuned) |  30.79 |  0.873 | 0.2119 | 0.2621 |  -0.91 |
| DLinear          | Local DL          |  31.52 |  0.89  | 0.3152 | 0.3899 |  -5.5  |
| TimesFM          | TSFM (zero-shot)  |  31.65 |  0.897 | 0.2478 | 0.3066 |  -8.37 |
| TiRex            | TSFM (zero-shot)  |  31.68 |  0.895 | 0.2475 | 0.3062 |  -8.87 |
| Chronos-2        | TSFM (zero-shot)  |  31.83 |  0.901 | 0.2199 | 0.272  |  -6.75 |
| Chronos-2 + mask | TSFM (zero-shot)  |  31.88 |  0.919 | 0.2211 | 0.2735 |  -7.82 |
| Moirai-2         | TSFM (zero-shot)  |  31.94 |  0.904 | 0.2492 | 0.3083 | -10.72 |
| TFT              | Local DL          |  31.94 |  0.855 | 0.3194 | 0.3952 | -10.08 |
| TFT              | Local DL          |  31.94 |  0.855 | 0.3194 | 0.3952 | -10.08 |
| TSB+EQ           | Statistical       |  32.35 |  0.903 | 0.2571 | 0.3181 |   1.27 |
| TiRex + mask     | TSFM (zero-shot)  |  32.36 |  0.937 | 0.2532 | 0.3133 | -11.62 |
| TFT              | Local DL          |  32.39 |  0.87  | 0.3239 | 0.4007 | -13.02 |
| Croston+EQ       | Statistical       |  32.5  |  0.906 | 0.2585 | 0.3198 |   1.45 |
| SBA+EQ           | Statistical       |  32.62 |  0.906 | 0.2592 | 0.3207 |   0.81 |
| Moirai-2 + mask  | TSFM (zero-shot)  |  32.62 |  0.937 | 0.2591 | 0.3206 | -10.4  |
| TSB              | Statistical       |  33.09 |  0.915 | 0.3309 | 0.4093 |  -6.95 |
| TimesFM + mask   | TSFM (zero-shot)  |  33.09 |  0.955 | 0.267  | 0.3303 |  -5.12 |
| ETS              | Statistical       |  33.2  |  0.924 | 0.332  | 0.4107 |  -2.03 |
| Croston          | Statistical       |  33.22 |  0.917 | 0.3322 | 0.411  |  -6.48 |
| ETS+EQ           | Statistical       |  33.49 |  0.933 | 0.2674 | 0.3308 |   5.48 |
| SBA              | Statistical       |  33.91 |  0.927 | 0.3391 | 0.4195 | -11.16 |
| TTM+EQ           | TSFM (zero-shot)  |  34.19 |  0.943 | 0.2712 | 0.3355 |   2.3  |
| TTM              | TSFM (zero-shot)  |  37.57 |  1.033 | 0.3757 | 0.4648 |  13.39 |
| sNaive           | Statistical       |  38.91 |  1.095 | 0.3891 | 0.4813 |  -0.77 |
| sNaive+EQ        | Statistical       |  40.87 |  1.149 | 0.3232 | 0.3998 |  10.41 |
| TTM + mask       | TSFM (zero-shot)  |  41.27 |  1.16  | 0.4127 | 0.5106 | -12.75 |

EQ = empirical residual quantiles added to a point forecaster; mask = stockout hours masked in the input context.
