"""E7-GPU (JORS R1, Major 7): 공통 워크로드에서 zero-shot 추론의 wall-clock과 피크 GPU 메모리.

run_zeroshot의 predictor를 그대로 호출하되 torch.cuda 피크 메모리를 기록한다.
메모리는 시계열 수에 거의 무관(배치 단위)하므로 느린 모델은 --limit로 줄여 측정한다.

사용: python scripts/measure_gpu_cost.py --model chronos2 --model-path ... [--limit N]
산출: results/gpu_cost/<model>.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.run_zeroshot import PREDICTORS, load_contexts, checkpoint_hash  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PREDICTORS))
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dataset", default="lt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()

    contexts, keys = load_contexts(a.dataset, False, None, a.limit)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()
    t0 = time.time()
    preds, levels = PREDICTORS[a.model](a.model_path, contexts, a.batch_size)
    torch.cuda.synchronize()
    sec = time.time() - t0
    rec = {"model": a.model, "checkpoint_hash": checkpoint_hash(ROOT / a.model_path),
           "dataset": a.dataset, "n_series": len(contexts), "horizon": 7,
           "wall_seconds": round(sec, 1),
           "seconds_per_10k_series": round(sec * 10000 / len(contexts), 1),
           "peak_gpu_mem_allocated_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "peak_gpu_mem_reserved_GB": round(torch.cuda.max_memory_reserved() / 2**30, 2),
           "baseline_allocated_GB": round(base_alloc / 2**30, 3),
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "batch_size": a.batch_size}
    out = ROOT / "results" / "gpu_cost"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.model}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
