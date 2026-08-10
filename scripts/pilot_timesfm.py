"""TimesFM-2.5 파일럿: timesfm 라이브러리(GitHub판)로 로컬 체크포인트 로드 + 소규모 예측."""

import os
import traceback

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
CKPT = "/home/ubuntu/tsfm-demand/data/checkpoints/google__timesfm-2.5-200m-pytorch"

try:
    import timesfm
    print("timesfm module:", timesfm.__file__, flush=True)
    print("attrs:", [a for a in dir(timesfm) if "2p5" in a or "TimesFM" in a], flush=True)
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(CKPT)
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024, max_horizon=64,
            normalize_inputs=True, use_continuous_quantile_head=True,
        )
    )
    ctx = [np.random.rand(500).astype(np.float32) * 3 for _ in range(4)]
    point, quantiles = model.forecast(horizon=7, inputs=ctx)
    print("point:", np.asarray(point).shape, "| quantiles:", np.asarray(quantiles).shape, flush=True)
    print("PILOT DONE", flush=True)
except Exception:
    traceback.print_exc()
