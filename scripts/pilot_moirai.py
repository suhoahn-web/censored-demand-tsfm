"""Moirai-2 파일럿: 로컬 체크포인트 로드 + GluonTS predictor로 소규모 예측 (moirai env)."""

import traceback

import numpy as np
import pandas as pd

CKPT = "/home/ubuntu/tsfm-demand/data/checkpoints/Salesforce__moirai-2.0-R-small"
HORIZON = 7

try:
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    module = Moirai2Module.from_pretrained(CKPT)
    model = Moirai2Forecast(
        module=module, prediction_length=HORIZON, context_length=512,
        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=8)
    print("predictor ready:", type(predictor).__name__, flush=True)

    data = ListDataset(
        [{"start": pd.Period("2024-01-01", freq="D"),
          "target": np.random.rand(300).astype(np.float32) * 3} for _ in range(4)],
        freq="D",
    )
    fcs = list(predictor.predict(data))
    f = fcs[0]
    print("forecast type:", type(f).__name__, flush=True)
    print("samples shape:", getattr(f, "samples", np.array([])).shape, flush=True)
    print("q0.5:", np.round(f.quantile(0.5), 3), flush=True)
    print("PILOT DONE", flush=True)
except Exception:
    traceback.print_exc()
