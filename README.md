[README (GitHub, ss-pi-stgnn-pv).md](https://github.com/user-attachments/files/34557458/README.GitHub.ss-pi-stgnn-pv.md)[Uplo# SS-PI-STGNN: label-free fault detection, degradation tracking and RUL for rooftop PV fleets

Code for the paper **"Unsupervised physics-informed GNN with fuzzy anomaly interpretation for PV fleet
degradation monitoring"** (N. Litvinov, I. Bizyarkin, V. Korkhov; submitted to *IEEE Access*).

SS-PI-STGNN is a modular, label-free framework for monitoring rooftop photovoltaic (PV) fleets. It
combines

- a **physics prior** of the expected power (irradiance, cell temperature, plant metadata);
- a **self-supervised spatio-temporal graph neural network** (SS-PI-STGNN) over a technically
  clustered station graph, trained without fault labels;
- interchangeable **degradation estimators** of the efficiency coefficient β1: batch
  **sign-perturbed sums (SPS)** or a recursive **Kalman filter** with the same interface;
- a **Mamdani fuzzy layer** that assigns physically meaningful fault types to detected events;
- **censoring-aware remaining useful life (RUL)** with explicit statuses (estimated, upper bound,
  censored, below threshold);
- two physics-based **baselines**: the performance-ratio rule (**PR**) and its fleet-referenced
  hourly extension (**PR+**, hour-of-day × month reference and technical peers).

All methods are compared on a **reproducible semi-synthetic benchmark**: faults, degradation and
configuration changes are injected into real data, network weights are **frozen** (SHA-256 checked
before and after every protocol), and detectors are compared **at an equal false-alarm budget**
with event-level cluster-bootstrap intervals and Holm-corrected paired permutation tests.

---

## Main findings (HKUST fleet, test year 2023)

- At the reference budget (1.19 alarm events per 1000 h), the network does **not** outperform the
  ratio baselines: overall event recall 0.083 against 0.120 (PR) and 0.139 (PR+), not
  significant. None of the seven pre-registered conditions for a gain of the network is met.
- The network is the best detector of **full-day outages** and detects faults on **newly
  commissioned plants** (recall 0.12 against 0 for PR and PR+). The fusion
  **PR+ OR STGNN-day** gives the highest recall (0.157) and F1 (0.266), so the network
  **complements** rather than replaces physics-based monitoring.
- The **Kalman filter** reduces the β1 error under linear degradation from 8.36 to 3.36 p.p. and
  the false degradation evidence from 0.333 to 0.111, but adapts to a permanent step only after
  about 232 days. **SPS** gives the most accurate plant-specific degradation rates. Both are biased
  by fleet-wide sensor steps.
- The **Mamdani layer** types outages reliably but confuses overgeneration with weather-driven
  deviations. The crisp version of the same rules types more detected events correctly.
- On the real fleet most stations are **censored** for RUL, and the sources rarely agree on which
  stations have a finite RUL (Cohen's κ ≤ 0.32).

See the paper for the full results, confidence intervals and limitations.

---

## Repository structure

```
.
├── pv-stgnn-fuzzy-multi-full.ipynb          # main notebook, sections 1–21 (see below)
├── run_pipeline.py                          # CLI: preprocessing → STGNN → SPS/Kalman → report
├── run_fuzzy_comparison.py                  # fuzzy-layer comparison and ablations on real data
├── requirements.txt
├── pv_*.py                                  # library modules (table below)
├── tests/                                   # synthetic fixture and tests (no HKUST data needed)
└── results/                                 # created at run time: cache, tables (.tex/.csv), figures
```

### Core pipeline

| Module | Purpose |
|---|---|
| `pv_physics_graph.py` | Physics model of expected power, availability masks, station graph, data signature |
| `pv_preprocess.py` | NaN-preserving hourly aggregation (`min_count`), comparison with zero filling |
| `pv_cohorts.py` | Coverage, commissioning cohorts, cold-start handling |
| `pv_clustering.py` | Multi-view technical clustering, weather regimes, anomaly-event features and classes |
| `pv_stgnn_unsupervised.py` | SS-PI-STGNN (baseline and multi-view), calibration of thresholds, event extraction |
| `pv_sps.py` | Sign-perturbed sums: quarterly β1 with exact confidence regions |
| `pv_kalman.py` | Kalman filter / RTS smoother for β0…β3 with the same API as `pv_sps.py` |
| `pv_estimator_comparison.py` | SPS vs Kalman on the final output: Silhouette, Davies–Bouldin, Calinski–Harabasz, ARI |
| `pv_fuzzy.py`, `pv_fuzzy_calibration.py` | Mamdani fuzzy layer (original and calibrated rule sets) |
| `pv_prplus.py` | PR+ baseline (hour × month reference, technical peers, causal rolling median) |
| `pv_report.py` | Figures and spreadsheets for the paper |

### Evaluation protocols (all optional and independent of the core pipeline)

| Protocol | Notebook section | Module | Question |
|---|---|---|---|
| P10 | 10 | `pv_semisynth_gen.py`, `pv_semisynth_eval.py` | Detection, fault type and β1 of the base detector on injected faults |
| P11 | 11 | `pv_semisynth_sensitivity.py` | Sensitivity to the quantile *q*, minimum event length and the source of η̂ |
| P12 | 12 | `pv_fixes_eval.py` | Detector refinements S0→S4 vs PR, PR+, zero-power rule, Isolation Forest |
| P13 | 13 | `pv_fcm_ablation.py` | Fuzzy c-means instead of k-means for weather regimes and technical clusters |
| P14 | 14 | `pv_stgnn_value.py` | SS-PI-STGNN vs PR/PR+ at equal budgets, strata, fusion, cold start, hard scenarios |
| P15 | 15 | `pv_modal_data.py`, `pv_semisynth_modal_gen.py`, `pv_stgnn_multimodal.py`, `pv_modal_eval.py` | Inverter-level modalities (STGNN-mm) |
| P16 | 16 | `pv_shading.py`, `pv_kalman_uncertainty.py`, `pv_stgnn_factorized.py` | Factorized multimodal STGNN, shading mask, Kalman uncertainty |
| — | 17 | `pv_metrics_summary.py`, `pv_metrics_stats.py` | Summary metrics, confidence intervals, paired tests |
| P18 | 18 | `pv_semisynth_config.py` | Configuration changes: module loss and module addition |
| P19 | 19 | `pv_rul.py` | RUL of the real fleet (PR fleet-adjusted, SPS/Kalman β1, STGNN η̂) |
| P20 | 20 | `pv_rul_semisynth.py` | Degradation rate and RUL against injected truth |
| — | 21 | `pv_metrics_stats.py` (`method_tables`) | Per-protocol method tables with best values and statistical ties |

Metrics are comparable **only within a protocol**: each protocol has its own events and its own
false-alarm budget.

---

## Installation

Python ≥ 3.10. A CUDA GPU is recommended for training (the full run was done on Kaggle); all
evaluation protocols also run on CPU with cached checkpoints.

```bash
git clone https://github.com/<user>/ss-pi-stgnn-pv.git
cd ss-pi-stgnn-pv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Install PyTorch for your CUDA version from <https://pytorch.org> if the default wheel does not fit.

---

## Data

The code uses the open **HKUST rooftop PV dataset**: 60 grid-connected rooftop stations
(2,230.8 kWp) on the campus of the Hong Kong University of Science and Technology, 2021–2023,
with on-site weather data and a Brick metadata model.

- Paper: Z. Lin et al., "A high-resolution three-year dataset supporting rooftop photovoltaics (PV)
  generation analytics", *Scientific Data* 12, 63 (2025),
  [doi:10.1038/s41597-025-04397-y](https://doi.org/10.1038/s41597-025-04397-y)
- Data: Dryad, [doi:10.5061/dryad.m37pvmd99](https://doi.org/10.5061/dryad.m37pvmd99)

Download and unpack the dataset. The notebook expects the original folder layout:

```
Dataset/
├── Metadata/PV generation system metadata.ttl
└── Time series dataset/
    ├── PV generation dataset/
    │   ├── PV stations without panel level optimizer/Site level dataset/*.csv
    │   └── PV stations with panel level optimizer/
    │       ├── Site level dataset/*.csv
    │       └── Inverter level dataset/*.csv        # only for sections 15–16
    └── Meteorological dataset/*.csv
```

Point the notebook to the folder that **contains** `Dataset/`, either with an environment variable
or by editing the default in the first configuration cell:

```bash
export PV_DATA_ROOT=/path/to/hkust     # folder with Dataset/ (default: data/hkust)
export PV_CODE_ROOT=/path/to/repo      # folder with pv_*.py (default: current directory)
```

The data are not redistributed in this repository.

---

## Quick start

### Notebook (recommended)

Set `PV_DATA_ROOT` (see [Data](#data)), start Jupyter from the repository root, open
`pv-stgnn-fuzzy-multi-full.ipynb` and run it top to bottom. The first configuration cell
resolves `DATA_ROOT` and `CODE_ROOT` and warns if `DATA_ROOT/Dataset` is missing. Key switches:

```python
ESTIMATOR   = "sps"                    # or "kalman"
train_seeds = [0, 1, 42, 77, 4700]     # five training seeds used throughout the paper
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
```

Sections 1–9 build the physics model, graph, clusters, train the STGNN, estimate β1 and compare SPS
with the Kalman filter. Sections 10–21 are the evaluation protocols; each one is optional, guarded
by its own import, and reads the frozen checkpoints of section 10. The notebook commentary, tables,
figures and LaTeX exports are in English.

### Command line

```bash
# full experiment (2 variants × 5 seeds), SPS or Kalman estimator
python run_pipeline.py --df_pickled hourly.pkl --ttl "PV generation system metadata.ttl" \
    --out results --epochs 15 --train_stride 1 --seeds 0 1 42 77 4700 --estimator sps

# individual protocols
python pv_semisynth_eval.py --help
python pv_semisynth_sensitivity.py --model_seeds 0 1 42 77 4700
```

`hourly.pkl` (or an `.xlsx` file passed with `--xlsx`) is the hourly table produced by the
preprocessing cells of the notebook (`data_preprocessing`, NaN-preserving aggregation), saved
e.g. with `df.to_pickle("hourly.pkl")`.

### Kaggle

Upload all `pv_*.py` files and `run_pipeline.py` as a Kaggle dataset, attach it together with the
HKUST dataset and set the two paths before the configuration cell:

```python
import os
os.environ["PV_DATA_ROOT"] = "/kaggle/input/<hkust-dataset>"
os.environ["PV_CODE_ROOT"] = "/kaggle/input/<code-dataset>"
```

Outputs are written to `results/` in the working directory (`/kaggle/working/results`).

---

## Reproducibility

- **Frozen weights.** After calibration the STGNN weights and thresholds are saved to
  `results/cache/stgnn_frozen_<variant>_<seed>.pt` before any online fine-tuning. All
  semi-synthetic protocols load these checkpoints with `requires_grad=False`; `fit` and
  `online_fine_tune` are blocked, and the SHA-256 of the weights is compared before and after each
  run.
- **Data signature.** Checkpoints store the station list and the SHA-256 of the training period.
  A checkpoint from another fleet or another preprocessing is never used silently: it is retrained
  or rejected with `CheckpointMismatch`.
- **Seeds.** Five training seeds × three placement seeds of the injected events. Seeds are not
  treated as independent observations: all rows of one injected event are resampled together.
- **Statistics.** Event-level cluster bootstrap (2000 replicates), paired permutation tests inside
  each event, Holm correction within a protocol; station bootstrap for degradation rates and RUL.
- **Equal budgets.** Thresholds are always set on data without injections; methods are compared at
  the same false-alarm rate on the clean test data.
- **Cache.** If preprocessing or the code of the detector changes, delete `results/cache/run_*.pkl`,
  the frozen checkpoints and the protocol caches (`results/semisynth|sensitivity|fixes|...`) and
  rerun the notebook.

---

## Outputs

Each protocol writes LaTeX tables (`booktabs`), CSV files and figures:

```
results/
├── cache/                 # frozen checkpoints and intermediate results
├── figures/               # fig_*.png
├── semisynth/  sensitivity/  fixes/  fcm/  value/  modal/  fmm/  config/  rul/  rul_semisynth/
└── summary/               # summary_metrics_all.tex, methods_<protocol>.tex, seed_check.tex
```

Wide tables are exported as `table*` for two-column layouts. In the per-protocol method tables, the
best value of a column is bold, and values not significantly different from the best (paired test,
Holm, p ≥ 0.05) are underlined.

---

## Tests

The tests run on a synthetic 30-station fleet (`tests/_semisynth_fixture.py`, three cohorts,
2021–2023) with effects unknown to the physics model (pyranometer drift, per-station model error and
shading, seasonal soiling); the HKUST data are not required.

```bash
python tests/synthetic_test.py              # SPS / Kalman
python tests/semisynth_test.py              # generation and evaluation with a small real STGNN (~2 min CPU)
python tests/sensitivity_test.py            # P11 reproduces P10 exactly
python tests/notebook_section12_13_test.py  # sections 10, 12–14 of the notebook
python tests/modal_test.py                  # P15 with synthetic inverters
python tests/fmm_test.py                    # P16
python tests/config_test.py                 # P18
python tests/rul_test.py                    # P19
python tests/rul_semisynth_test.py          # P20
python tests/summary_test.py                # section 17
python tests/method_tables_test.py          # section 21
```

---

## Citation

If you use this code, please cite the paper (details will be updated after publication):

```bibtex
@article{litvinov2026sspistgnn,
  author  = {Litvinov, Nikolai and Bizyarkin, Iaroslav and Korkhov, Vladimir},
  title   = {Unsupervised physics-informed {GNN} with fuzzy anomaly interpretation for {PV} fleet
             degradation monitoring},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Submitted}
}
```

and the dataset:

```bibtex
@article{lin2025hkust,
  author  = {Lin, Zinan and Zhou, Qi and Wang, Zhe and Wang, Ce and Bookhart, Davis Boyd and
             Leung-Shea, Marcus},
  title   = {A high-resolution three-year dataset supporting rooftop photovoltaics ({PV})
             generation analytics},
  journal = {Scientific Data},
  volume  = {12},
  pages   = {63},
  year    = {2025},
  doi     = {10.1038/s41597-025-04397-y}
}
```

---

## Acknowledgements

This work was supported by the Ministry of Economic Development of the Russian Federation
(Agreement No. 139-15-2025-007, dated April 16, 2025; ID: 000000C313925P3O0002).
Saint Petersburg University, Saint Petersburg, Russian Federation.

## License

To be added (e.g., MIT for the code). The HKUST dataset is distributed under its own license on
Dryad.

## Contact

Nikolai N. Litvinov — n.n.litvinov@spbu.ru
ading README (GitHub, ss-pi-stgnn-pv).md…]()
