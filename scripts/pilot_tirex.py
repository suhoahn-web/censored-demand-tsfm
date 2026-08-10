"""TiRex 파일럿: 로컬 체크포인트 로드 + forecast API 확인 + 소규모 예측 (lab06, detached 실행용)."""

import inspect
import os

import numpy as np
import torch

# tirex load_model은 "org/model" 형태만 허용 → 심링크 구조 + 상대경로로 우회
os.environ["HF_HUB_OFFLINE"] = "1"
CKPT_ROOT = "/home/ubuntu/tsfm-demand/data/checkpoints"
os.makedirs(f"{CKPT_ROOT}/NX-AI", exist_ok=True)
if not os.path.exists(f"{CKPT_ROOT}/NX-AI/TiRex"):
    # from_pretrained은 ckpt "파일" 경로를 기대 → 파일에 심링크
    os.symlink(f"{CKPT_ROOT}/NX-AI__TiRex/model.ckpt", f"{CKPT_ROOT}/NX-AI/TiRex")
os.chdir(CKPT_ROOT)

from tirex import load_model  # noqa: E402

# xLSTM CUDA 커널 컴파일(nvcc 함정) 회피 — torch 백엔드 사용
model = load_model("NX-AI/TiRex", device="cuda", backend="torch")
print("loaded:", type(model).__name__, flush=True)
print("forecast sig:", inspect.signature(model.forecast), flush=True)

ctx = [np.random.rand(500).astype(np.float32) * 3 for _ in range(8)]
out = model.forecast(context=ctx, prediction_length=7)
if isinstance(out, tuple):
    for i, o in enumerate(out):
        print(f"out[{i}]:", type(o).__name__, getattr(o, "shape", None), flush=True)
else:
    print("out:", type(out).__name__, getattr(out, "shape", None), flush=True)

# chronos 호환 확인 (torch 다운그레이드 영향)
from chronos import BaseChronosPipeline  # noqa: E402
pipe = BaseChronosPipeline.from_pretrained(
    "/home/ubuntu/tsfm-demand/data/checkpoints/amazon__chronos-2",
    device_map="cuda", torch_dtype=torch.bfloat16)
q, _ = pipe.predict_quantiles([torch.tensor(c) for c in ctx[:2]], prediction_length=7,
                              quantile_levels=[0.1, 0.5, 0.9])
print("chronos still ok:", q[0].shape, flush=True)
print("PILOT DONE", flush=True)
