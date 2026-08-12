# Foundation models for censored fresh-retail demand — reproduction package

Code and configuration for the experiments reported in *"Forecasting for perishable
replenishment under demand censoring: an evaluation of time series foundation
models."*

Everything in this repository regenerates from public data: the two FreshRetailNet
releases (CC-BY-4.0) and published model checkpoints. No proprietary data is used.

## 1. What the pipeline produces

| Stage | Entry point | Output |
|---|---|---|
| Daily panels | `src/data/prepare_daily.py` | `data/processed/{lt,frn50k}_daily_{train,eval}.parquet` |
| Demand-pattern classes | `src/eval/stratified.py` (auto) | `data/processed/{ds}_classification.parquet` |
| Zero-shot forecasts | `src/experiments/run_zeroshot.py` | `results/zeroshot/*.parquet` (+ `.meta.json`) |
| Fine-tuning | `src/experiments/run_finetune_chronos2.py` | `results/finetune_v2/<run>/predictions.parquet` |
| Classical baselines | `src/experiments/run_traditional.py` | `results/zeroshot/{croston,sba,tsb,ets,snaive}_*.parquet` |
| Empirical quantiles | `src/experiments/add_empirical_quantiles.py` | `results/zeroshot/*eq_*.parquet` |
| Gradient boosting | `src/experiments/run_lightgbm.py` | `results/zeroshot/lightgbm_*.parquet` |
| Local DL (reproduced) | `scripts/integrate_local_dl.py` | `results/zeroshot/{tft,dlinear}_*.parquet` |
| Accuracy metrics | `src/eval/evaluate_runs.py` | `results/summary_metrics.csv` |
| Inventory simulation | `src/eval/inventory_sim.py` | `results/inventory_*.csv` |
| Accuracy–cost analysis | `src/eval/accuracy_vs_inventory.py` | `results/kendall_tau_*.csv` |
| Statistical tests | `src/eval/stat_tests.py` | `results/mcb_*.csv`, `results/nemenyi_*.csv` |
| Figures / tables | `src/viz/make_figures.py`, `src/viz/make_tables.py` | `results/figures/`, `results/tables/` |

Every forecast file is written with a sibling `.meta.json` recording the model,
checkpoint hash, dataset, censoring treatment, quantile levels, seed and library
versions, so any number in the paper can be traced back to the run that produced it.

## 2. Environments

The model families have incompatible dependency pins, so each runs in its own
environment. Create them with conda:

```bash
# 1) foundation models: Chronos-2, TiRex, TTM, TimesFM-2.5
conda create -n tsfm python=3.11 -y
conda run -n tsfm pip install torch pandas pyarrow "chronos-forecasting>=2.0" \
    accelerate peft tirex-ts granite-tsfm
conda run -n tsfm pip install -e external/timesfm[torch]   # GitHub release, not PyPI

# 2) Moirai-2 (pins torch 2.4)
conda create -n moirai python=3.11 -y && conda run -n moirai pip install uni2ts

# 3) reproduction of the published TFT / DLinear / recovery baselines
conda create -n repro python=3.10 -y
conda run -n repro pip install -r external/frn-50k-baseline/requirements.txt
conda run -n repro pip install "setuptools<81"   # pytorch-lightning 1.9 needs pkg_resources

# 4) analysis only (CPU): metrics, inventory simulation, figures, tables
pip install pandas pyarrow statsforecast lightgbm scipy scikit-posthocs \
    matplotlib adjustText tabulate
```

Known pitfalls, each of which cost us a debugging cycle:

- **TimesFM 2.5** is not on PyPI under that version; install the GitHub package and
  load with `timesfm.TimesFM_2p5_200M_torch.from_pretrained(<local path>)`.
- **TiRex** accepts only `"org/model"` identifiers, so a local checkpoint must be
  reached through a symlink named `NX-AI/TiRex` pointing at `model.ckpt`, with the
  working directory set to the checkpoint root.
- **pytorch-lightning 1.9** imports `pkg_resources`, removed in setuptools ≥ 81.
- Installing `uni2ts` downgrades torch; keep it in its own environment.

## 3. Reproducing the results

```bash
# data
python -m src.data.prepare_daily --dataset lt
python -m src.data.prepare_daily --dataset frn50k

# forecasts (GPU); repeat per model and condition, see scripts/queue_*.sh
python -m src.experiments.run_zeroshot --model chronos2 \
    --model-path data/checkpoints/amazon__chronos-2 --dataset lt \
    --out results/zeroshot/chronos2_lt_raw.parquet
python -m src.experiments.run_finetune_chronos2 --mode lora --rank 16 --seed 0 \
    --out results/finetune_v2/lora_r16_s0

# baselines (CPU)
python -m src.experiments.run_traditional --dataset lt
python -m src.experiments.add_empirical_quantiles --dataset lt
python -m src.experiments.run_lightgbm --dataset lt

# evaluation and outputs (CPU)
python -m src.eval.evaluate_runs
python -m src.eval.inventory_sim --dataset lt
python -m src.eval.stat_tests --dataset lt
python -m src.eval.accuracy_vs_inventory --dataset lt
python -m src.viz.make_figures --dataset lt
python -m src.viz.make_tables
```

The `scripts/queue_*.sh` files run the full grids unattended and log each run's
success or failure, which is how the reported experiments were executed.

## 4. Design decisions that affect the numbers

Four choices affect how the results should be read:

1. **Scoring excludes censored days.** Accuracy is computed only on evaluation days
   with zero stocked-out hours, following the benchmark's reference
   implementation. Scoring on all days hides the censoring bias.
2. **Point forecasters receive empirical quantiles.** Without this, an inventory
   comparison measures whether a method has a predictive distribution rather than
   how good its forecast is.
3. **All-missing contexts are restored.** Masking stocked-out days empties the
   context of series that were out of stock throughout the training window (88 of
   50,000 in the 50K release); those series keep their observed values, and the
   count is recorded in each run's metadata.
4. **Recovered-demand runs are scored against observed sales.** Training data
   differs across censoring treatments; the evaluation target never does.

## 5. Repository layout

```
├── src/
│   ├── data/         panel construction
│   ├── experiments/  forecasting runs (zero-shot, fine-tuning, baselines)
│   ├── eval/         metrics, inventory simulation, statistical tests
│   └── viz/          figures and tables
├── scripts/          unattended run queues, baseline patches, validators
├── configs/          experiment configuration
├── external/         upstream baseline code (frn-50k-baseline, timesfm)
└── results/          metrics, statistical tests, figures, tables
```

`scripts/verify_draft_numbers.py` recomputes every number reported in the paper
from the stored results and reports any mismatch; it is the quickest way to check
that the manuscript and this repository agree.

`scripts/patch_frn_baseline.py` documents the two modifications made to the
upstream baseline code: replacing Hugging Face downloads with local parquet reads
(the compute node has no access to the HF CDN) and saving predictions in addition
to aggregate metrics. No modelling logic is changed.

## 6. Compute

All GPU work ran on a single NVIDIA A100 80GB. Indicative costs: zero-shot
inference over 10,000 series takes seconds to tens of minutes depending on the
model; one Chronos-2 fine-tuning run takes about five minutes; reproducing the
published TFT baseline takes about one hour.

## 7. License and citation

Code is released under the MIT License (see `LICENSE`). The FreshRetailNet
datasets are distributed by Dingdong-Inc under CC-BY-4.0 and are not
redistributed here.

This repository is archived on Zenodo:
**[10.5281/zenodo.21902351](https://doi.org/10.5281/zenodo.21902351)** (release
v1.0.1). Please cite both the software and the accompanying paper:

> Ahn, S. (2026). *censored-demand-tsfm: reproduction package* [Software].
> Zenodo. https://doi.org/10.5281/zenodo.21902351

> Ahn, S. Forecasting for perishable replenishment under demand censoring: an
> evaluation of time series foundation models. (Under review.)
