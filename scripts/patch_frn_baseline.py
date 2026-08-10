"""frn-50k-baseline의 HF load_dataset 호출을 로컬 parquet 로드로 패치 (lab06용).

lab06은 HF CDN 차단 → load_dataset("Dingdong-Inc/FreshRetailNet-50K") 불가.
서버 작업 사본만 수정하며, 원본 리포/로컬 사본은 건드리지 않는다.
실행: python scripts/patch_frn_baseline.py ~/tsfm-demand/external/frn-50k-baseline
"""

import sys
from pathlib import Path

RAW = "~/tsfm-demand/data/raw/FreshRetailNet-50K/data"

TRAIN_OLD = """        dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
        df = dataset['train'].to_pandas()"""
TRAIN_NEW = f"""        import os as _os
        df = pd.read_parquet(_os.path.expanduser("{RAW}/train.parquet"))"""

PRED_OLD = """    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    df_train = dataset['train'].to_pandas()
    df_eval = dataset['eval'].to_pandas()"""
PRED_NEW = f"""    import os as _os
    df_train = pd.read_parquet(_os.path.expanduser("{RAW}/train.parquet"))
    df_eval = pd.read_parquet(_os.path.expanduser("{RAW}/eval.parquet"))"""


# latent_demand_recovery 공통 패턴 (generate_data.py 1곳, app.py 2곳, 4칸 들여쓰기)
GEN_OLD = """    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    data = dataset['train'].to_pandas()"""
GEN_NEW = f"""    import os as _os
    data = pd.read_parquet(_os.path.expanduser("{RAW}/train.parquet"))"""


def patch(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new.strip() in text:
        print(f"already patched: {path}")
        return
    if old not in text:
        raise SystemExit(f"patch target not found in {path}")
    path.write_text(text.replace(old, new))
    print(f"patched: {path} ({text.count(old)} site(s))")


if __name__ == "__main__":
    repo = Path(sys.argv[1]).expanduser()
    tft = repo / "demand_forecasting" / "TFT"
    patch(tft / "trainTFT.py", TRAIN_OLD, TRAIN_NEW)
    patch(tft / "predictTFT.py", PRED_OLD, PRED_NEW)
    rec = repo / "latent_demand_recovery" / "exp"
    patch(rec / "data" / "generate_data.py", GEN_OLD, GEN_NEW)
    patch(rec / "app.py", GEN_OLD, GEN_NEW)
    # predictTFT: 지표만 저장하던 것을 예측값(merged_df)도 저장하도록 (재고 시뮬레이션용)
    patch(
        tft / "predictTFT.py",
        """    df_pred = get_pred(dataset, df_eval, data_type, start_epoch=-1, verbose=True)
    df_res = get_metrics(df_pred, ["psd>=0", "psd>=1", "psd<1"])""",
        """    df_pred = get_pred(dataset, df_eval, data_type, start_epoch=-1, verbose=True)
    for (_dt, _dfp) in df_pred:
        _dfp.to_parquet(f"./{_dt}_predictions.parquet")
    df_res = get_metrics(df_pred, ["psd>=0", "psd>=1", "psd<1"])""",
    )
