"""TTM(granite-tsfm) + TimesFM-2.5(transformers) 파일럿: 로컬 로드 + 소규모 예측 (lab06 detached용)."""

import os
import traceback

import numpy as np
import torch

os.environ["HF_HUB_OFFLINE"] = "1"
CKPT_ROOT = "/home/ubuntu/tsfm-demand/data/checkpoints"

print("=== TTM pilot ===", flush=True)
try:
    from tsfm_public import TinyTimeMixerForPrediction
    ttm = TinyTimeMixerForPrediction.from_pretrained(
        f"{CKPT_ROOT}/ibm-granite__granite-timeseries-ttm-r2").to("cuda").eval()
    cfg = ttm.config
    print("TTM loaded. context_length:", cfg.context_length,
          "| prediction_length:", cfg.prediction_length, flush=True)
    x = torch.rand(4, cfg.context_length, 1, device="cuda")
    with torch.no_grad():
        out = ttm(past_values=x)
    print("TTM out:", out.prediction_outputs.shape, flush=True)
except Exception:
    traceback.print_exc()

print("=== TimesFM-2.5 pilot ===", flush=True)
try:
    from transformers import TimesFmModelForPrediction
    tfm = TimesFmModelForPrediction.from_pretrained(
        f"{CKPT_ROOT}/google__timesfm-2.5-200m-pytorch",
        torch_dtype=torch.bfloat16).to("cuda").eval()
    ctx = [torch.rand(500, device="cuda") * 3 for _ in range(4)]
    with torch.no_grad():
        out = tfm(past_values=ctx, freq=[0] * 4, return_dict=True)
    print("TimesFM out keys:", list(out.keys()) if hasattr(out, "keys") else type(out), flush=True)
    mp = getattr(out, "mean_predictions", None)
    fp = getattr(out, "full_predictions", None)
    print("mean:", getattr(mp, "shape", None), "| full:", getattr(fp, "shape", None), flush=True)
except Exception:
    traceback.print_exc()

print("PILOT DONE", flush=True)
