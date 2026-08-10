"""실험 결과 일괄 평가: 예측 parquet들 → 지표 표 (WAPE/WPE/MASE/WQL/CRPS).

공식 eval 구간 예측만 평가한다 (rolling origin 예측은 cutoff별 정답 구성 후 별도 평가).
사용: python -m src.eval.evaluate_runs
"""

import json
from pathlib import Path

import pandas as pd

from src.eval.metrics import evaluate_forecast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"
KEY = ["store_id", "product_id"]


def load_truth(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth = pd.read_parquet(PROCESSED / f"{dataset}_daily_eval.parquet",
                            columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    insample = pd.read_parquet(PROCESSED / f"{dataset}_daily_train.parquet",
                               columns=KEY + ["dt", "sale_amount"])
    return truth, insample


def main() -> None:
    truths: dict = {}

    # 복원수요(frn50krec)로 학습·예측했더라도 채점은 항상 관측 판매(50K) 기준
    TRUTH_OF = {"frn50krec": "frn50k"}

    def get_truth(ds: str):
        ds = TRUTH_OF.get(ds, ds)
        if ds not in truths:
            truths[ds] = load_truth(ds)
        return truths[ds]

    rows = []

    # zero-shot: 공식 origin만 (cutoff 없는 것)
    for meta_path in sorted((RESULTS / "zeroshot").glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("cutoff_date"):
            continue
        pred = pd.read_parquet(str(meta_path).replace(".meta.json", ""))
        truth, insample = get_truth(meta["dataset"])
        for nostock in [False, True]:
            m = evaluate_forecast(pred, truth, insample, nostock_only=nostock)
            rows.append({"run": meta_path.stem.replace(".parquet.meta", ""),
                         "adapt": "zero-shot", "dataset": meta["dataset"],
                         "mask": meta["mask_stockout"],
                         "eval_filter": "nostock" if nostock else "all", **m})

    # fine-tune (v1은 시드 미반영 결함으로 참고용, 논문에는 v2만 사용 — decisions.md)
    for ft_dir, ver in [("finetune", "v1"), ("finetune_v2", "v2")]:
        base = RESULTS / ft_dir
        if not base.exists():
            continue
        for run_dir in sorted(base.iterdir()):
            meta_file = run_dir / "meta.json"
            pred_file = run_dir / "predictions.parquet"
            if not (meta_file.exists() and pred_file.exists()):
                continue
            meta = json.loads(meta_file.read_text())
            pred = pd.read_parquet(pred_file)
            truth, insample = get_truth(meta["dataset"])
            adapt = f"lora-r{meta['rank']}" if meta["mode"] == "lora" else "full-ft"
            lr = meta.get("learning_rate")
            for nostock in [False, True]:
                m = evaluate_forecast(pred, truth, insample, nostock_only=nostock)
                rows.append({"run": f"{ver}/{run_dir.name}", "ver": ver, "adapt": adapt,
                             "dataset": meta["dataset"], "mask": meta["mask_stockout"],
                             "seed": meta["seed"], "lr": lr,
                             "eval_filter": "nostock" if nostock else "all",
                             "train_sec": meta["train_seconds"], **m})

    df = pd.DataFrame(rows)
    out = RESULTS / "summary_metrics.csv"
    df.to_csv(out, index=False)

    # 시드 평균 요약
    df["ver"] = df.get("ver", pd.Series(["-"] * len(df))).fillna("-")
    df["lr"] = df.get("lr", pd.Series([None] * len(df)))
    g = (df.groupby(["dataset", "eval_filter", "ver", "adapt", "lr", "mask"], dropna=False)
         .agg(wape=("wape", "mean"), wape_sd=("wape", "std"),
              wpe=("wpe", "mean"), mase=("mase_mean", "mean"),
              wql=("wql", "mean"), crps=("crps", "mean"), n=("wape", "size"))
         .round(4).reset_index())
    print(g.to_string(index=False))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
