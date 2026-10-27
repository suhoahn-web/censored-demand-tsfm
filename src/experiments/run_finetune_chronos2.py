"""실험 2: Chronos-2 LoRA/full 파인튜닝 → 공식 eval 예측 저장.

사용 예 (lab06):
  python -m src.experiments.run_finetune_chronos2 --mode lora --rank 8 --seed 0 \
      --out results/finetune/c2_lora_r8_s0

- 학습 입력: LT train 구간(≤2025-07-13)의 eval 대상 시계열 — 시간 기반 분할로 미래 누출 없음 (절대 규칙 4)
- 학습 후 공식 eval 컨텍스트로 분위수 예측 저장 (zero-shot과 동일 포맷 → 동일 평가 코드)
- 시드·체크포인트 해시 기록 (절대 규칙 3, 5)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.experiments.run_zeroshot import (HORIZON, PROJECT_ROOT, QUANTILES,
                                          checkpoint_hash, load_contexts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="lora", choices=["lora", "full"])
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank (mode=lora)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="lt", choices=["lt", "frn50k", "frn50krec"])
    ap.add_argument("--model-path", default="data/checkpoints/amazon__chronos-2")
    ap.add_argument("--num-steps", type=int, default=1000)
    ap.add_argument("--learning-rate", type=float, default=None,
                    help="미지정 시 chronos 기본값 사용")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--mask-stockout", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="파일럿용 시계열 제한")
    ap.add_argument("--out", required=True, help="출력 디렉토리 (results/finetune/...)")
    ap.add_argument("--cutoff-date", default=None,
                    help="(R1 E2) origin별 재학습: 컨텍스트를 이 날짜까지로 자르고 다음 7일을 예측")
    ap.add_argument("--min-context-days", type=int, default=None,
                    help="(R1 E2) 이력이 N일 미만인 시계열 제외 (zero-shot origin 실행과 동일 집합)")
    ap.add_argument("--out-parquet", default=None,
                    help="(R1 E2) zero-shot 포맷으로 results/zeroshot/<name>.parquet(+meta) 저장")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    contexts, keys = load_contexts(args.dataset, args.mask_stockout, args.cutoff_date,
                                   args.limit, min_context_days=args.min_context_days)
    print(f"train series: {len(contexts)}", flush=True)

    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(
        PROJECT_ROOT / args.model_path, device_map="cuda", torch_dtype=torch.bfloat16)

    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_kwargs = dict(prediction_length=HORIZON, finetune_mode=args.mode,
                      num_steps=args.num_steps, batch_size=args.batch_size,
                      output_dir=out_dir / "ckpt",
                      seed=args.seed)  # HF Trainer 시드 — 절대 규칙 5 (시드 반복)
    if args.mode == "lora":
        fit_kwargs["lora_config"] = {"r": args.rank}
        # chronos 기본 lr 1e-6은 LoRA에 과소 — LoRA 표준 1e-4 기본 적용 (decisions.md)
        fit_kwargs["learning_rate"] = args.learning_rate if args.learning_rate else 1e-4
    elif args.learning_rate is not None:
        fit_kwargs["learning_rate"] = args.learning_rate

    t0 = time.time()
    ft_pipe = pipe.fit([torch.tensor(c) for c in contexts], **fit_kwargs)
    train_sec = time.time() - t0
    print(f"fit done in {train_sec:.0f}s", flush=True)

    # 공식 eval 예측 (zero-shot과 동일 포맷)
    out = []
    for i in range(0, len(contexts), args.batch_size):
        q_list, _ = ft_pipe.predict_quantiles(contexts[i:i + args.batch_size],
                                              prediction_length=HORIZON,
                                              quantile_levels=QUANTILES)
        out.append(torch.stack([t.squeeze(0) for t in q_list]).float().cpu().numpy())
    preds = np.concatenate(out, axis=0)

    import pandas as pd
    rows = keys.loc[keys.index.repeat(HORIZON)].reset_index(drop=True)
    rows["h"] = np.tile(np.arange(1, HORIZON + 1), len(keys))
    for j, q in enumerate(QUANTILES):
        rows[f"q{q}"] = preds[:, :, j].reshape(-1)
    rows.to_parquet(out_dir / "predictions.parquet", index=False)

    n_trainable = sum(p.numel() for p in ft_pipe.model.parameters() if p.requires_grad)
    meta = {"mode": args.mode, "rank": args.rank if args.mode == "lora" else None,
            "seed": args.seed, "dataset": args.dataset,
            "mask_stockout": args.mask_stockout, "num_steps": args.num_steps,
            "learning_rate": fit_kwargs.get("learning_rate"), "batch_size": args.batch_size,
            "base_checkpoint_hash": checkpoint_hash(PROJECT_ROOT / args.model_path),
            "train_seconds": round(train_sec), "n_series": len(contexts),
            "n_trainable_params": n_trainable, "torch": torch.__version__}
    meta.update({"cutoff_date": args.cutoff_date, "min_context_days": args.min_context_days})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    if args.out_parquet:  # R1 E2: multi_origin_* 스크립트가 읽는 zero-shot 포맷
        op = PROJECT_ROOT / args.out_parquet
        op.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(op, index=False)
        zmeta = {"model": f"chronos2-{args.mode}" + (f"-r{args.rank}" if args.mode == "lora" else ""),
                 "model_path": args.model_path, "checkpoint_hash": meta["base_checkpoint_hash"],
                 "dataset": args.dataset, "mask_stockout": args.mask_stockout,
                 "cutoff_date": args.cutoff_date, "min_context_days": args.min_context_days,
                 "quantiles": QUANTILES, "horizon": HORIZON, "n_series": len(contexts),
                 "seed": args.seed, "learning_rate": fit_kwargs.get("learning_rate"),
                 "num_steps": args.num_steps, "train_seconds": round(train_sec),
                 "torch": torch.__version__}
        Path(str(op) + ".meta.json").write_text(json.dumps(zmeta, indent=2))
        print("saved:", op, flush=True)
    print("saved:", out_dir, flush=True)


if __name__ == "__main__":
    main()
