# -*- coding: utf-8 -*-
"""R1 second pass: shelf life 3 with lead time 1 (separates the protection interval from the shelf life)."""
import pandas as pd, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
CELLS = {"low_margin_waste_dominant": (0.3, 0.5), "medium_margin": (1.0, 0.7), "high_margin_stockout_dominant": (3.0, 0.9)}
def load(tag):
    x = pd.read_csv(R / f"inventory_{tag}.csv"); x = x[x.is_fixed_cr & ~x.run.str.contains("equ_")]
    return x
for tag in ["lt_sl2_L0", "lt_sl3_L0", "lt_sl2_L1_indep", "lt_sl2_L1_comon", "lt_sl3_L1_indep", "lt_sl3_L1_comon"]:
    try: x = load(tag)
    except FileNotFoundError: print(tag, "missing"); continue
    print(f"\n=== {tag}: n runs {x.run.nunique()} ===")
    for sc, (b, q) in CELLS.items():
        s = x[x.scenario == sc].sort_values("cost_per_unit_demand").reset_index(drop=True)
        rk = lambda r: (int(s.index[s.run == r][0]) + 1) if (s.run == r).any() else None
        top = ", ".join(f"{r.run}({r.cost_per_unit_demand:.4f})" for r in s.head(3).itertuples())
        print(f"  b={b} q={q}: top3 {top}; LightGBM rank {rk('lightgbm_lt_raw')} ({s.set_index('run').cost_per_unit_demand.get('lightgbm_lt_raw', np.nan):.4f}), "
              f"LoRA lr1e-3 rank {rk('ft_lora_r16_lr1e-3_s0')} ({s.set_index('run').cost_per_unit_demand.get('ft_lora_r16_lr1e-3_s0', np.nan):.4f}), "
              f"fill q=0.9 range {s.fill_rate.min():.3f}-{s.fill_rate.max():.3f}" if q == 0.9 else
              f"  b={b} q={q}: top3 {top}; LightGBM rank {rk('lightgbm_lt_raw')}, LoRA lr1e-3 rank {rk('ft_lora_r16_lr1e-3_s0')}")
