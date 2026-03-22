# AIMS: Adaptive and Intelligent Management of Slicing for Next-Generation ITS Networks

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**AIMS** is a framework for classifying how network slicing policies impact Intelligent Transportation Systems (ITS) applications. It combines centralized machine learning, federated learning, and security analysis to evaluate vehicular QoS classification under realistic deployment conditions.

The framework supports three complementary experiment modes:

| Mode | Models | Approach |
|---|---|---|
| **Centralized** | RF, CatBoost, TabNet | HPO with Optuna (TPE), GroupKFold CV |
| **Federated** | DNN, LSTM, GRU | FedAvg/FedProx, IID/NonIID via Flower simulation |
| **Security** | DNN, LSTM, GRU | Label-flip & gradient scaling attacks; Krum, TrimmedMean defenses |

## Dataset

Vehicular network QoS data from **158 vehicles** over **450 seconds**, with 4 application classes (Safety, Efficiency, Entertainment, Generic) and 25 engineered features from RTT, PDR, and Throughput metrics.

Based on: T. do Vale Saraiva et al., "An Application-Driven Framework for Intelligent Transportation Systems Using 5G Network Slicing," IEEE TITS, vol. 22, no. 8, 2021 ([saraivacode/framework_its_sdn](https://github.com/saraivacode/framework_its_sdn)).

## Installation

```bash
python --version  # 3.12+
pip install -r requirements.txt
```

For strict reproducibility, use the experiment-specific frozen environment files instead (see [Environments](#environments)).

## Quick Start

```bash
cd code

# Centralized models (reference config)
python main.py --compare --experiment-id "centralized-v1" --n-trials 15 --n-trials-tabnet 40

# Federated learning only (no RF/TabNet/CatBoost)
python main.py --federated-only --experiment-id "fl-v1"

# Security experiments only (no RF/TabNet/CatBoost)
python main.py --security-only --security-sensitivity-epochs --experiment-id "sec-v1"

# Everything at once
python main.py --federated --security --security-sensitivity-epochs \
    --compare --experiment-id "full-run"
```

## Experiment Isolation

Every run creates an isolated output directory under `results/`. If `--experiment-id` is omitted, it defaults to a UTC timestamp.

```
results/
├── centralized-v1/          ← --experiment-id "centralized-v1"
│   ├── random_forest/
│   ├── tabnet/
│   ├── catboost/
│   └── model_comparison.{png,csv}
├── fl-v1/                   ← --experiment-id "fl-v1"
│   └── federated/
└── 20260321T143052Z/        ← default (UTC timestamp)
```

## CLI Reference

### Global Options

| Option | Default | Description |
|---|---|---|
| `--experiment-id` | UTC timestamp | Subdirectory name under `--results-dir` |
| `--results-dir` | `../results` | Base results directory |
| `--csv` | `../data/aims_dataset.csv` | Dataset path |
| `--random-state` | 42 | Reproducibility seed |

### Centralized Models

| Option | Default | Description |
|---|---|---|
| `--n-trials` | 40 | Optuna trials for RF and CatBoost |
| `--n-trials-tabnet` | 40 | Optuna trials for TabNet |
| `--n-splits` | 5 | GroupKFold CV folds |
| `--skip-rf` / `--skip-tabnet` / `--skip-catboost` | — | Skip individual models |
| `--compare` | — | Generate comparison report after training |
| `--test-comparison-only` | — | Re-generate comparison without re-training |

### Federated Learning

| Option | Default | Description |
|---|---|---|
| `--federated` | — | Run FL + centralized NN baselines + RF/TabNet/CatBoost |
| `--federated-only` | — | FL + centralized NN baselines only (no RF/TabNet/CatBoost) |
| `--fl-rounds` | 30 | Communication rounds |
| `--fl-local-epochs` | 3 | Local training epochs per round |
| `--fl-clients` | 3 | Simulated FL clients (RSUs) |
| `--fl-strategies` | `FedAvg FedProx` | Aggregation strategies |
| `--fl-distributions` | `IID NonIID` | Data distribution modes |
| `--fl-models` | `DNN LSTM GRU` | Neural architectures |
| `--skip-centralized` | — | Skip centralized NN baselines |

### FL Security

| Option | Default | Description |
|---|---|---|
| `--security` | — | Phase 1a (label-flip) + Phase 1b (label-flip + gradient scaling) |
| `--security-only` | — | Same as `--security`, skips RF/TabNet/CatBoost |
| `--skip-phase1a` / `--skip-phase1b` | — | Skip individual phases |
| `--security-sensitivity-epochs` | — | Vary local epochs (1, 3, 5, 10) under attack |

## Impact Classification

| Level | Label | Description |
|---|---|---|
| 0 | Adequate | All QoS requirements met |
| 1 | Warning | Slight degradation, non-critical apps affected |
| 2 | Severe | Significant degradation, multiple apps impacted |
| 3 | Critical | Severe degradation, safety applications at risk |

## Environments

This project was validated across two environments depending on the experiment set.

### Centralized experiments — RTX PRO 4500

Reference files in [`docs/environment_AIMS/`](docs/environment_AIMS/):

| File | Content |
|---|---|
| `requirements_rtxpro4500_ch6_working.txt` | Frozen pip packages |
| `env_rtxpro4500_ch6_ok.txt` | Environment summary |
| `nvidia_smi_rtxpro4500_ch6.txt` | GPU/driver info |

### Federated and security experiments — NVIDIA L40S

Reference files in [`docs/environment_AIMS_FL_SEC/`](docs/environment_AIMS_FL_SEC/):

| File | Content |
|---|---|
| `requirements_l40s_working.txt` | Frozen pip packages |
| `env_l40s_ok.txt` | Environment summary |
| `nvidia_smi_l40s.txt` | GPU/driver info |

Validated core stack: `torch==2.4.1+cu124`, `tensorflow==2.18.0`, `protobuf==5.29.6`.

### Reproducibility Notes

The root `requirements.txt` lists logical project dependencies. For strict reproducibility, prefer the experiment-specific frozen files above. Each `docs/environment_*/` directory also includes a `rebuild_env_*.sh` script for recreating the environment from scratch.

## Project Structure

```
AIMS/
├── code/
│   ├── main.py                    # Entry point (all experiment modes)
│   ├── preprocess_dataset.py      # Rolling-window features, outlier clipping
│   ├── impact_labeling.py         # QoS-to-impact level assignment
│   ├── train_model_rf.py          # Random Forest pipeline
│   ├── train_model_tabnet.py      # TabNet pipeline
│   ├── train_model_catboost.py    # CatBoost pipeline
│   ├── compare_results.py         # Model comparison report
│   ├── results_manager.py         # Results collection and JSON export
│   ├── save_utils.py              # Artifact saving utilities
│   └── federated/
│       ├── fl_main.py             # FL + security orchestrator
│       ├── fl_config.py           # Defaults and NonIID allocation
│       ├── fl_data.py             # Data loading, IID/NonIID partitioning
│       ├── fl_models.py           # DNN, LSTM, GRU builders (Keras)
│       ├── fl_server.py           # Flower simulation, custom strategies
│       ├── fl_client.py           # AIMSFlowerClient (NumPyClient)
│       ├── fl_centralized.py      # Centralized NN baselines
│       ├── fl_security.py         # Attacks, defenses, C2A metric
│       ├── fl_results.py          # FL results manager (JSON, CSV, LaTeX)
│       └── fl_visualizations.py   # Convergence/comparison/security plots
├── data/
│   └── aims_dataset.csv
├── docs/
│   ├── AIMS_options_ref_v3_en.md  # Complete options and artifact reference
│   ├── environment_AIMS/          # Centralized frozen environment
│   └── environment_AIMS_FL_SEC/   # FL/security frozen environment
├── results/                       # Experiment outputs (per experiment-id)
├── requirements.txt               # Logical dependencies
├── requirements_rtxpro4500_ch6_working.txt
└── requirements_l40s_working.txt
```

## Detailed Reference

For experiment composition tables, per-artifact inventory, and full directory tree, see **[docs/AIMS_options_ref_v3_en.md](docs/AIMS_options_ref_v3_en.md)**.

## Citation

```bibtex
@article{saraiva2025aims,
  title={AIMS: Adaptive and Intelligent Management of Slicing for Next-Generation ITS Networks},
  author={Saraiva, Tiago do Vale},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Federal University of the State of Rio de Janeiro (UNIRIO)
- Dataset based on [saraivacode/framework_its_sdn](https://github.com/saraivacode/framework_its_sdn), which uses [Mininet-WiFi](https://github.com/intrig-unicamp/mininet-wifi), [Ryu SDN Controller](https://osrg.github.io/ryu/), and [SUMO](https://sumo.dlr.de/docs/Installing.html)

## Contact

- **Tiago do Vale Saraiva** — [tiago.saraiva@uniriotec.br](mailto:tiago.saraiva@uniriotec.br)
