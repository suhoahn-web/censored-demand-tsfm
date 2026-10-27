"""실험 1: TSFM zero-shot 일 단위 7일 예측.

사용 예 (lab06):
  python -m src.experiments.run_zeroshot --model chronos2 \
      --model-path data/checkpoints/amazon__chronos-2 \
      --dataset lt --out results/zeroshot/chronos2_lt_raw.parquet

- 컨텍스트: 각 eval 시계열의 train 구간 일 판매량 (--cutoff-date로 rolling origin)
- --mask-stockout: 품절 시간이 있는 날을 NaN 처리 (검열 마스킹 조건)
- --limit N: 파일럿용 시계열 수 제한
- 출력: series key × horizon(1..7) × 분위수 + 메타데이터(체크포인트 해시 — 절대 규칙 3)
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
QUANTILES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
HORIZON = 7
KEY = ["store_id", "product_id"]


def checkpoint_hash(path: Path) -> str:
    """체크포인트 디렉토리의 파일명+크기 기반 해시 (실험 로그용)."""
    items = sorted(f"{p.name}:{p.stat().st_size}" for p in path.rglob("*") if p.is_file())
    return hashlib.sha256("|".join(items).encode()).hexdigest()[:16]


def load_contexts(dataset: str, mask_stockout: bool, cutoff_date: str | None,
                  limit: int | None, max_context_days: int | None = None,
                  min_context_days: int | None = None) -> tuple[list, pd.DataFrame]:
    train = pd.read_parquet(PROCESSED / f"{dataset}_daily_train.parquet",
                            columns=KEY + ["dt", "sale_amount", "stock_hour6_22_cnt"])
    eval_keys = (pd.read_parquet(PROCESSED / f"{dataset}_daily_eval.parquet", columns=KEY)
                 .drop_duplicates().sort_values(KEY).reset_index(drop=True))
    if limit:
        eval_keys = eval_keys.head(limit)
    train = train.merge(eval_keys, on=KEY)
    if cutoff_date:
        train = train[train["dt"] <= pd.Timestamp(cutoff_date)]
    if min_context_days:
        # 이른 origin에서는 아직 개점하지 않은 시계열이 있어 컨텍스트가 비거나 너무 짧다.
        # 모델이 붕괴하지 않도록 최소 이력 길이를 요구한다 (몇 개가 빠졌는지 메타에 기록).
        n_obs = train.groupby(KEY)["dt"].size()
        keep = n_obs[n_obs >= min_context_days].index
        train = train.set_index(KEY).loc[keep].reset_index()
        print(f"  min-context {min_context_days}d: kept {len(keep)} of "
              f"{len(n_obs)} series", flush=True)
    train = train.sort_values(KEY + ["dt"])
    orig = None
    if mask_stockout:
        orig = train.groupby(KEY, sort=True)["sale_amount"].apply(
            lambda s: s.to_numpy(dtype=np.float32))
        train.loc[train["stock_hour6_22_cnt"] > 0, "sale_amount"] = np.nan
    grouped = train.groupby(KEY, sort=True)
    contexts = [g["sale_amount"].to_numpy(dtype=np.float32) for _, g in grouped]
    n_restored = 0
    if mask_stockout:
        # 전 기간 품절이면 마스킹 후 컨텍스트가 전부 NaN이 되어 모델이 붕괴한다
        # (TimesFM은 예외, Chronos-2는 무의미한 값 산출). 이런 시계열은 관측값 유지.
        orig_list = list(orig)
        for i, c in enumerate(contexts):
            if np.isnan(c).all():
                contexts[i] = orig_list[i]
                n_restored += 1
        if n_restored:
            print(f"  all-NaN after masking -> restored observed: {n_restored} series",
                  flush=True)
    load_contexts.n_restored = n_restored
    if max_context_days:  # 이력 길이 효과 분리용 (LT를 50K 길이로 절단)
        contexts = [c[-max_context_days:] for c in contexts]
    keys_sorted = pd.DataFrame(list(grouped.groups.keys()), columns=KEY)
    return contexts, keys_sorted


def predict_chronos2(model_path: str, contexts: list, batch_size: int) -> np.ndarray:
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        model_path, device_map="cuda", torch_dtype=torch.bfloat16)
    out = []
    for i in range(0, len(contexts), batch_size):
        batch = contexts[i:i + batch_size]
        q_list, _ = pipe.predict_quantiles(batch, prediction_length=HORIZON,
                                           quantile_levels=QUANTILES)
        q = torch.stack([t.squeeze(0) for t in q_list])  # (B, HORIZON, n_q)
        out.append(q.float().cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"  {i + len(batch)}/{len(contexts)}", flush=True)
    return np.concatenate(out, axis=0), QUANTILES


def predict_tirex(model_path: str, contexts: list, batch_size: int):
    """TiRex (tirex-ts). load_model이 'org/model' 형태만 받아 파일 심링크로 우회.
    분위수는 모델 기본 9개(0.1~0.9)."""
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    root = PROJECT_ROOT / "data" / "checkpoints"
    (root / "NX-AI").mkdir(exist_ok=True)
    link = root / "NX-AI" / "TiRex"
    if not link.exists():
        link.symlink_to(root / "NX-AI__TiRex" / "model.ckpt")
    cwd = os.getcwd()
    os.chdir(root)
    from tirex import load_model
    model = load_model("NX-AI/TiRex", device="cuda", backend="torch")
    os.chdir(cwd)
    q, _mean = model.forecast(context=[torch.tensor(c) for c in contexts],
                              batch_size=batch_size, prediction_length=HORIZON)
    levels = [round(0.1 * k, 1) for k in range(1, 10)]
    return q.float().cpu().numpy(), levels


def predict_ttm(model_path: str, contexts: list, batch_size: int):
    """TTM r2 (granite-tsfm): 점예측 전용, context 512 고정 → 좌측 0패딩+관측마스크,
    horizon 96 중 앞 7일 사용. 분위수는 [0.5] 단일(점예측)로 저장."""
    from tsfm_public import TinyTimeMixerForPrediction
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # CPU 폴백 (JORS R1 E3)
    model = TinyTimeMixerForPrediction.from_pretrained(model_path).to(dev).eval()
    ctx_len = model.config.context_length
    out = []
    for i in range(0, len(contexts), batch_size):
        batch = contexts[i:i + batch_size]
        x = np.zeros((len(batch), ctx_len), dtype=np.float32)
        m = np.zeros((len(batch), ctx_len), dtype=np.float32)
        for j, c in enumerate(batch):
            c = c[-ctx_len:]
            obs = ~np.isnan(c)
            x[j, -len(c):] = np.where(obs, c, 0.0)
            m[j, -len(c):] = obs.astype(np.float32)
        with torch.no_grad():
            o = model(past_values=torch.tensor(x).unsqueeze(-1).to(dev),
                      past_observed_mask=torch.tensor(m).unsqueeze(-1).to(dev))
        out.append(o.prediction_outputs[:, :HORIZON, 0].float().cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"  {i + len(batch)}/{len(contexts)}", flush=True)
    preds = np.concatenate(out)[:, :, None]  # (N, 7, 1)
    return preds, [0.5]


def predict_timesfm(model_path: str, contexts: list, batch_size: int):
    """TimesFM-2.5 (timesfm GitHub판). 출력 (B,H,10) = [mean, q0.1..q0.9] → 분위수 9개만 저장."""
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_path)
    model.compile(timesfm.ForecastConfig(
        max_context=1024, max_horizon=64,
        normalize_inputs=True, use_continuous_quantile_head=True))
    out = []
    for i in range(0, len(contexts), batch_size):
        batch = [c[-1024:] for c in contexts[i:i + batch_size]]
        _point, q = model.forecast(horizon=HORIZON, inputs=batch)
        out.append(np.asarray(q)[:, :, 1:10])
        if (i // batch_size) % 20 == 0:
            print(f"  {i + len(batch)}/{len(contexts)}", flush=True)
    levels = [round(0.1 * k, 1) for k in range(1, 10)]
    return np.concatenate(out), levels


def predict_moirai(model_path: str, contexts: list, batch_size: int):
    """Moirai-2 (uni2ts, moirai env 전용). GluonTS predictor → QuantileForecast.
    단변량 전용(2.0에서 공변량 지원 제거)."""
    import pandas as pd_
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(model_path),
        prediction_length=HORIZON, context_length=512,
        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    predictor = model.create_predictor(batch_size=batch_size)
    ds = ListDataset(
        [{"start": pd_.Period("2000-01-01", freq="D"), "target": c} for c in contexts],
        freq="D")
    levels = [round(0.1 * k, 1) for k in range(1, 10)]
    out = []
    for n, f in enumerate(predictor.predict(ds)):
        out.append(np.stack([f.quantile(q) for q in levels], axis=-1))
        if n % 2000 == 0:
            print(f"  {n}/{len(contexts)}", flush=True)
    return np.stack(out), levels


PREDICTORS = {"chronos2": predict_chronos2, "tirex": predict_tirex,
              "ttm": predict_ttm, "timesfm": predict_timesfm,
              "moirai2": predict_moirai}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(PREDICTORS))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k", "frn50krec"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--mask-stockout", action="store_true")
    ap.add_argument("--cutoff-date", default=None,
                    help="rolling origin용 컨텍스트 마지막 날짜 (YYYY-MM-DD)")
    ap.add_argument("--limit", type=int, default=None, help="파일럿용 시계열 수 제한")
    ap.add_argument("--max-context-days", type=int, default=None,
                    help="컨텍스트를 최근 N일로 절단 (이력 길이 효과 분리)")
    ap.add_argument("--min-context-days", type=int, default=None,
                    help="이력이 N일 미만인 시계열 제외 (이른 rolling origin용)")
    args = ap.parse_args()

    torch.manual_seed(0)
    contexts, keys = load_contexts(args.dataset, args.mask_stockout,
                                   args.cutoff_date, args.limit,
                                   args.max_context_days, args.min_context_days)
    lens = [len(c) for c in contexts]
    print(f"series: {len(contexts)}, context len min/med/max: "
          f"{min(lens)}/{int(np.median(lens))}/{max(lens)}", flush=True)

    preds, levels = PREDICTORS[args.model](args.model_path, contexts, args.batch_size)

    rows = keys.loc[keys.index.repeat(HORIZON)].reset_index(drop=True)
    rows["h"] = np.tile(np.arange(1, HORIZON + 1), len(keys))
    for j, q in enumerate(levels):
        rows[f"q{q}"] = preds[:, :, j].reshape(-1)
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out_path, index=False)

    meta = {"model": args.model, "model_path": args.model_path,
            "checkpoint_hash": checkpoint_hash(PROJECT_ROOT / args.model_path),
            "dataset": args.dataset, "mask_stockout": args.mask_stockout,
            "cutoff_date": args.cutoff_date, "quantiles": levels,
            "max_context_days": args.max_context_days,
            "min_context_days": args.min_context_days,
            "n_all_nan_restored": getattr(load_contexts, "n_restored", 0),
            "horizon": HORIZON, "n_series": len(contexts),
            "torch": torch.__version__, "seed": 0}
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print("saved:", out_path, flush=True)


if __name__ == "__main__":
    main()
