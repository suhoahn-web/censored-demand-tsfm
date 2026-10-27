"""P0 Task 2: EQ 분해를 사전지정 분위수에서 공정하게 재계산한다.

배경: 기존 §5.6 요약은 변형(C/D)별로 테스트 시뮬레이션에서 비용 최소 q를
골랐다(idxmin) — 테스트셋 q 최적화. A/B는 분위수가 q=0.5 하나뿐이라 영향이 없었지만
C/D는 영향을 받았다. 여기서는:

  1) q=0.5 공통 셀: 네 변형 모두 정의. 구조적 항등성(A==C, B==D)을 검증한다.
  2) 사전지정 분위수(§4.5의 고정 분위수 설계: b=0.3->q=0.5, b=1->q=0.7, b=3->q=0.9):
     A/B는 유일한 운영점(중앙값), C/D는 사전지정 q에서 평가한다.
  3) Gain_C = (Cost_A - Cost_C) / (Cost_A - Cost_D)를 셀별로 보고한다.
     b=0.3 셀에서 Gain_C = 0은 구조적(중앙값 발주에는 스프레드가 무의미).

산출:
  results/eq_ablation_common_q_results.csv
  results/eq_ablation_q_selection_audit.md
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
PRESPECIFIED_Q = {0.3: 0.5, 1.0: 0.7, 3.0: 0.9}
REPORT_Q = [0.5, 0.7, 0.9]
VARIANTS = ["A_point", "B_recenter", "C_spread", "D_fullEQ"]

eq = pd.read_csv(R / "eq_ablation_lt.csv")

# ---------- 1) 공통 q 격자 결과 (지시서 최소 셀: q in {0.5,0.7,0.9} x b in {1,3}) ----------
keep_cols = ["model", "variant", "q", "b", "cost_per_unit_demand", "fill_rate",
             "lost_rate", "waste_rate", "avg_inventory", "wape", "wpe", "wql",
             "cov0.7", "cov0.9"]
common = eq[eq.q.isin(REPORT_Q)][keep_cols].copy()
common["operating_note"] = np.where(
    common.variant.isin(["A_point", "B_recenter"]),
    "single operating point (median); rows exist only at q=0.5", "")
common.to_csv(R / "eq_ablation_common_q_results.csv", index=False)
print(f"common-q rows: {len(common)}")

# ---------- 2) 구조적 항등성 검증 ----------
at5 = eq[np.isclose(eq.q, 0.5)]
pv5 = at5.pivot_table(index=["model", "b"], columns="variant",
                      values="cost_per_unit_demand")
ident_AC = float(np.max(np.abs(pv5["A_point"] - pv5["C_spread"])))
ident_BD = float(np.max(np.abs(pv5["B_recenter"] - pv5["D_fullEQ"])))
print(f"identity at q=0.5: max|A-C|={ident_AC:.2e}, max|B-D|={ident_BD:.2e}")

# ---------- 3) 사전지정 분위수에서의 분해 ----------
lines = []
for b, q in PRESPECIFIED_Q.items():
    s = eq[np.isclose(eq.b, b)]
    a = s[(s.variant == "A_point") & np.isclose(s.q, 0.5)].set_index("model")
    bb = s[(s.variant == "B_recenter") & np.isclose(s.q, 0.5)].set_index("model")
    c = s[(s.variant == "C_spread") & np.isclose(s.q, q)].set_index("model")
    d = s[(s.variant == "D_fullEQ") & np.isclose(s.q, q)].set_index("model")
    ca, cb2 = a.cost_per_unit_demand.mean(), bb.cost_per_unit_demand.mean()
    cc, cd = c.cost_per_unit_demand.mean(), d.cost_per_unit_demand.mean()
    gain_c = (ca - cc) / (ca - cd) if ca - cd > 1e-9 else np.nan
    gain_b = (ca - cb2) / (ca - cd) if ca - cd > 1e-9 else np.nan
    lines.append(dict(b=b, q_prespec=q, cost_A=ca, cost_B=cb2, cost_C=cc,
                      cost_D=cd, gain_share_B=gain_b, gain_share_C=gain_c))
    print(f"b={b} q={q}: A={ca:.4f} B={cb2:.4f} C={cc:.4f} D={cd:.4f} "
          f"| Gain_B={gain_b:.3f} Gain_C={gain_c:.3f}")
summary = pd.DataFrame(lines)
summary.to_csv(R / "eq_ablation_prespecified_summary.csv", index=False)

# ---------- 4) 감사 문서 ----------
audit = f"""# EQ 분해 q 선택 감사 (P0 Task 2)

**Was test-set q optimization used: YES** (기존 §5.6 요약 수치에 한해)

## 무엇이 문제였나

`src/experiments/eq_ablation.py`의 원래 요약 블록은

```python
best = s.loc[s.groupby(["model", "variant"])["cost_per_unit_demand"].idxmin()]
```

로 변형·모델별 **테스트 시뮬레이션 비용을 최소화하는 q**를 골랐다. A/B 변형은
분위수가 q=0.5 하나뿐이어서 영향을 받지 않았지만, C/D는 전체 q 격자에서 최소
비용 셀이 선택되었다. 이는 지시서의 invalid 사례 5(테스트 구간 비용 최소화)에
해당한다.

## 무엇으로 바꿨나

사전지정 규칙: **본문 §4.5의 고정 분위수 설계**(Huber et al. 2019의 고정 목표
서비스수준 방식, 원 설계 문서에 명시) — b=0.3→q=0.5, b=1.0→q=0.7, b=3.0→q=0.9.
검증셋 선택이 아니라 분석적으로 사전지정된 운영 규칙이다. A/B는 중앙값이 유일한
운영점이므로 q=0.5에서 평가하고, C/D가 추가 운영점을 여는 것 자체를 결과의 일부로
보고한다 (지시서의 "Preferred handling").

## 구조적 항등성 (q=0.5)

시뮬레이션 결과로 검증: max|Cost_A − Cost_C| = {ident_AC:.2e},
max|Cost_B − Cost_D| = {ident_BD:.2e}. 중앙값에서 발주하면 스프레드는 주문량을
바꾸지 않으므로 A≡C, B≡D. **스프레드의 가치는 중앙값 위에서 발주할 때만 생긴다.**

## 수정된 분해 (모델 5개 평균, 사전지정 분위수)

| b | q(사전지정) | A(점) | B(재중심) | C(스프레드) | D(전체) | 재중심 기여 | 스프레드 기여 |
|---|---|---|---|---|---|---|---|
"""
for r in lines:
    gb = "–" if np.isnan(r["gain_share_B"]) else f"{r['gain_share_B']:.0%}"
    gc = "–" if np.isnan(r["gain_share_C"]) else f"{r['gain_share_C']:.0%}"
    audit += (f"| {r['b']:g} | {r['q_prespec']:g} | {r['cost_A']:.4f} | "
              f"{r['cost_B']:.4f} | {r['cost_C']:.4f} | {r['cost_D']:.4f} | "
              f"{gb} | {gc} |\n")
audit += """
기여도 = (Cost_A − Cost_X) / (Cost_A − Cost_D). b=0.3 셀은 사전지정 q가 0.5라
A≡C, B≡D — 스프레드 기여 0은 구조적이다.

## 기존 주장과의 차이

- 기존: "C가 비용 감소의 96–98%를 설명" (테스트셋 최소 q 기준).
- 수정: 스프레드 기여는 **운영 분위수에 따라 0%(중앙값) → 78%(q=0.7) →
  98%(q=0.9)로 증가**한다. b=3.0 셀은 테스트 최소 q가 사전지정 q(0.9)와 일치해
  수치가 그대로이고, b=1.0 셀은 96% → 78%로 낮아졌다.
- 원고 §5.6, 초록, 기여 3, §6.2, 결론을 이 수치로 재작성했다. 스프레드 기여가
  분위수와 함께 커진다는 사실은 RQ1의 속성 이동과 같은 메커니즘의 다른 단면이다.

## 산출물

- `results/eq_ablation_common_q_results.csv` — q∈{0.5,0.7,0.9}×b∈{0.3,1,3} 전 셀,
  비용·fill·lost·waste·평균재고·WAPE·WPE·WQL·coverage.
- `results/eq_ablation_prespecified_summary.csv` — 위 표.
- Fig. 6·Table 11은 사전지정 분위수 기준으로 재생성.
"""
(R / "eq_ablation_q_selection_audit.md").write_text(audit, encoding="utf-8")
print("wrote eq_ablation_q_selection_audit.md")
