"""실험 4-A: 뉴스벤더 평가 — 분위수 예측 F̂⁻¹(CR)을 발주량으로.

프로토콜 (설계서 v2.0 6절; Theodorou et al. 2025 + Ban & Rudin 2019 근거):
  - critical ratio CR ∈ {0.5, 0.7, 0.8, 0.9, 0.95}, 발주량 = 해당 분위수 예측 (0 클리핑)
  - 비용: co=1 정규화, cu = CR/(1-CR) → cost = co·max(q-d,0) + cu·max(d-q,0)
  - 지표: 평균비용, fill rate(β), 폐기비율(과잉발주/총발주)
  - 채점: 원 논문 공식 프로토콜(nostock: 품절 없는 날만 — 관측판매≈수요) 기본
  - 분위수 미보유 CR은 인접 분위수 선형보간 (보간 불가 시 해당 CR 생략)

사용: python -m src.eval.newsvendor --dataset lt
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]
CRS = [0.5, 0.7, 0.8, 0.9, 0.95]


TRUTH_OF = {"frn50krec": "frn50k"}  # 복원수요 학습이어도 채점은 관측 판매 기준


def load_truth(dataset: str) -> pd.DataFrame:
    dataset = TRUTH_OF.get(dataset, dataset)
    t = pd.read_parquet(PROCESSED / f"{dataset}_daily_eval.parquet",
                        columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    t = t.sort_values(KEY + ["dt"]).reset_index(drop=True)
    t["h"] = t.groupby(KEY).cumcount() + 1
    return t


def order_at_cr(pred: pd.DataFrame, cr: float) -> np.ndarray | None:
    """CR 분위수 발주량 — 보유 분위수에서 (필요시 선형보간으로) 추출."""
    levels = sorted(float(c[1:]) for c in pred.columns if c.startswith("q"))
    if cr in levels:
        return np.clip(pred[f"q{cr}"].to_numpy(), 0, None)
    lo = [l for l in levels if l < cr]
    hi = [l for l in levels if l > cr]
    if not lo or not hi:
        return None
    l0, l1 = max(lo), min(hi)
    w = (cr - l0) / (l1 - l0)
    q = (1 - w) * pred[f"q{l0}"].to_numpy() + w * pred[f"q{l1}"].to_numpy()
    return np.clip(q, 0, None)


def eval_run(pred: pd.DataFrame, truth: pd.DataFrame, nostock: bool = True) -> list[dict]:
    merged = pred.merge(truth[KEY + ["h", "sale_amount", "stock_hour6_22_cnt"]],
                        on=KEY + ["h"], how="inner")
    if nostock:
        merged = merged[merged["stock_hour6_22_cnt"] == 0]
    d = merged["sale_amount"].to_numpy()
    rows = []
    for cr in CRS:
        q = order_at_cr(merged, cr)
        if q is None:
            continue
        cu = cr / (1 - cr)
        over = np.maximum(q - d, 0)
        under = np.maximum(d - q, 0)
        rows.append({
            "cr": cr,
            "mean_cost": float((over + cu * under).mean()),
            "fill_rate": float(np.minimum(q, d).sum() / d.sum()),
            "waste_frac": float(over.sum() / max(q.sum(), 1e-9)),
            "n_obs": len(d),
        })
    return rows


def collect_predictions(dataset: str):
    """zeroshot 및 finetune_v2의 공식 origin 예측을 (이름, df) 목록으로."""
    items = []
    for meta_path in sorted((RESULTS / "zeroshot").glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("cutoff_date") or meta["dataset"] != dataset:
            continue
        name = meta_path.name.replace(".parquet.meta.json", "")
        if name.startswith("ctxabl_"):
            continue
        if re.search(r"_s\d+_raw$", name):
            continue  # 시드 복제본은 Table 7 집계 전용  # 맥락길이 ablation은 본 분석 제외
        items.append((name, pd.read_parquet(str(meta_path).replace(".meta.json", ""))))
    ft = RESULTS / "finetune_v2"
    if ft.exists():
        for run_dir in sorted(ft.iterdir()):
            mf, pf = run_dir / "meta.json", run_dir / "predictions.parquet"
            if not (mf.exists() and pf.exists()):
                continue
            meta = json.loads(mf.read_text())
            if meta["dataset"] != dataset or meta.get("seed") != 0:
                continue  # 뉴스벤더 1차는 seed 0 대표
            items.append((f"ft_{run_dir.name}", pd.read_parquet(pf)))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k", "frn50krec"])
    ap.add_argument("--all-days", action="store_true", help="nostock 필터 해제(민감도)")
    args = ap.parse_args()

    truth = load_truth(args.dataset)
    rows = []
    for name, pred in collect_predictions(args.dataset):
        for r in eval_run(pred, truth, nostock=not args.all_days):
            rows.append({"run": name, **r})
    df = pd.DataFrame(rows)
    out = RESULTS / f"newsvendor_{args.dataset}{'_alldays' if args.all_days else ''}.csv"
    df.to_csv(out, index=False)

    pivot = df.pivot_table(index="run", columns="cr", values="mean_cost").round(4)
    print("=== 뉴스벤더 평균비용 (행=모델, 열=CR) ===")
    print(pivot.to_string())
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
