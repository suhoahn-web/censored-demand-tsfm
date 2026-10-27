**Decomposing the empirical-quantile procedure: forecast metrics and inventory cost for the four variants of Section 4.3, averaged over the five point forecasters, at the prespecified ordering quantile of each cost regime.**

| Variant | WAPE | WPE | WQL | cov. at 0.9 | cost, b=0.3 (q=0.5) | cost, b=1 (q=0.7) | cost, b=3 (q=0.9) |
|---|---|---|---|---|---|---|---|
| A. point forecast | 27.86 | -7.98 | 0.2786 | – | 0.0700 | 0.1953 | 0.5531 |
| B. recentre only | 26.91 | -1.93 | 0.2691 | – | 0.0602 | 0.1566 | 0.4321 |
| C. spread only (median preserved) | 27.86 | -7.98 | 0.2280 | 0.882 | 0.0700 | 0.1255 | 0.1701 |
| D. full empirical quantiles | 26.91 | -1.93 | 0.2214 | 0.901 | 0.0602 | 0.1063 | 0.1619 |

*WAPE and WPE in per cent. Variants A and B carry no quantiles above the median and are therefore evaluated at their only operating point, q = 0.5; variants C and D are evaluated at the prespecified quantile of each regime. At q = 0.5 the orders of C equal those of A and the orders of D equal those of B exactly, so the b = 0.3 column shows the structural identity. Variant C leaves the point forecast unchanged, so its WAPE and WPE equal those of variant A by construction. No quantile is selected on the test data.*
