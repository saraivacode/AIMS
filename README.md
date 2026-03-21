# AIMS: Adaptive and Intelligent Management of Slicing for Next-Generation ITS Networks

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**AIMS** is a framework for classifying how network slicing policies impact Intelligent Transportation Systems (ITS) applications. It combines centralized machine learning, federated learning, and security analysis to evaluate vehicular QoS classification under realistic deployment conditions.

The framework supports three research directions, each mapped to a thesis chapter:

| Chapter | Topic | Models | Approach |
|---|---|---|---|
| **Ch. 6** | Centralized classification | RF, TabNet, CatBoost | HPO with Optuna, GroupKFold CV |
| **Ch. 7** | Federated classification | DNN, LSTM, GRU | FedAvg/FedProx, IID/NonIID |
| **Ch. 8** | FL security analysis | DNN, LSTM, GRU | FedAvg/FedProx under label-flip & gradient scaling; Krum, TrimmedMean defenses |

## Dataset

Vehicular network QoS data from **158 vehicles** over **450 seconds**, with 4 application classes (Safety, Efficiency, Entertainment, Generic) and 15+ engineered features from RTT, PDR, and Throughput metrics.

Based on: T. do Vale Saraiva et al., "An Application-Driven Framework for Intelligent Transportation Systems Using 5G Network Slicing," IEEE TITS, vol. 22, no. 8, 2021 ([saraivacode/framework_its_sdn](https://github.com/saraivacode/framework_its_sdn)).

## Installation

```bash
python --version  # 3.12+
pip install -r requirements.txt
```

## Quick Start

```bash
cd code

# Ch. 6 — Train all centralized models and compare (primary config)
python main.py --compare --experiment-id "ch6-iscc" --n-trials 15 --n-trials-tabnet 40

# Ch. 7 — Run federated learning experiments (no RF/TabNet/CatBoost)
python main.py --federated-only --experiment-id "fl-v1"

# Ch. 8 — Run all security experiments (no RF/TabNet/CatBoost)
python main.py --security-only --security-sensitivity-epochs --experiment-id "security-v1"

# Everything at once
python main.py --federated --security --security-sensitivity-epochs \
    --compare --experiment-id "full-run"
```

## Experiment Isolation

Every run creates an isolated output directory. Results never overwrite each other:

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
    └── ...
```

If `--experiment-id` is omitted, it defaults to a UTC timestamp (e.g., `20260321T143052Z`).

## CLI Reference

### Global Options

| Option | Default | Description |
|---|---|---|
| `--experiment-id` | UTC timestamp | Experiment identifier (subdirectory under `--results-dir`) |
| `--results-dir` | `../results` | Base results directory |
| `--csv` | `../data/aims_dataset.csv` | Dataset path |
| `--random-state` | 42 | Reproducibility seed |

### Centralized Models (Ch. 6)

```bash
python main.py --compare --n-trials 15 --n-trials-tabnet 40
```

| Option | Default | Description |
|---|---|---|
| `--n-trials` | 40 | Optuna trials for RF and CatBoost |
| `--n-trials-tabnet` | 40 | Optuna trials for TabNet |
| `--n-splits` | 5 | GroupKFold CV folds |
| `--skip-rf` | — | Skip Random Forest |
| `--skip-tabnet` | — | Skip TabNet |
| `--skip-catboost` | — | Skip CatBoost |
| `--compare` | — | Generate comparison report after training |
| `--test-comparison-only` | — | Re-generate comparison without re-training |

### Federated Learning (Ch. 7)

```bash
python main.py --federated-only --fl-rounds 30
```

| Option | Default | Description |
|---|---|---|
| `--federated` | — | Run FL + centralized NN baselines + RF/TabNet/CatBoost |
| `--federated-only` | — | Run FL + centralized NN baselines only (no RF/TabNet/CatBoost) |
| `--fl-rounds` | 30 | Communication rounds |
| `--fl-local-epochs` | 3 | Local training epochs per round |
| `--fl-clients` | 3 | Simulated FL clients (RSUs) |
| `--fl-strategies` | `FedAvg FedProx` | Aggregation strategies |
| `--fl-distributions` | `IID NonIID` | Data distribution modes |
| `--fl-models` | `DNN LSTM GRU` | Neural architectures |
| `--skip-centralized` | — | Skip centralized NN baselines |

### FL Security (Ch. 8)

```bash
python main.py --security-only --security-sensitivity-epochs
```

| Option | Default | Description |
|---|---|---|
| `--security` | — | Run Phase 1a (label-flip) + Phase 1b (label-flip + gradient scaling) with FedAvg/FedProx + Krum/TrimmedMean |
| `--security-only` | — | Same as `--security` but skips RF/TabNet/CatBoost (implies `--skip-rf --skip-tabnet --skip-catboost`) |
| `--skip-phase1a` | — | Skip Phase 1a when using `--security` |
| `--skip-phase1b` | — | Skip Phase 1b when using `--security` |
| `--security-sensitivity-epochs` | — | Vary local epochs (1, 3, 5, 10) under attack |

## Impact Classification

The framework assigns impact levels using application-specific QoS thresholds:

| Level | Label | Description |
|---|---|---|
| 0 | Adequate | All QoS requirements met |
| 1 | Warning | Slight degradation, non-critical apps affected |
| 2 | Severe | Significant degradation, multiple apps impacted |
| 3 | Critical | Severe degradation, safety applications at risk |

## Project Structure

```
AIMS/
├── code/
│   ├── main.py                    # Main orchestrator (all experiment modes)
│   ├── train_model_rf.py          # Random Forest pipeline
│   ├── train_model_tabnet.py      # TabNet pipeline
│   ├── train_model_catboost.py    # CatBoost pipeline
│   ├── preprocess_dataset.py      # Data preprocessing
│   ├── impact_labeling.py         # QoS-to-impact level assignment
│   ├── save_utils.py              # Artifact saving utilities
│   ├── results_manager.py         # Results collection and JSON export
│   ├── compare_results.py         # Model comparison report
│   └── federated/
│       ├── fl_main.py             # FL + security experiment orchestrator
│       ├── fl_config.py           # FL defaults and configuration
│       ├── fl_data.py             # Data loading, IID/NonIID partitioning
│       ├── fl_models.py           # DNN, LSTM, GRU model builders (Keras)
│       ├── fl_server.py           # FedAvg/FedProx aggregation engine
│       ├── fl_client.py           # Client training simulation
│       ├── fl_centralized.py      # Centralized NN baselines
│       ├── fl_security.py         # Attacks (label-flip, gradient scaling),
│       │                          #   defenses (Krum, TrimmedMean), C2A metric
│       ├── fl_results.py          # FL results manager (JSON, CSV, LaTeX)
│       └── fl_visualizations.py   # Convergence, comparison, security plots
├── data/
│   └── aims_dataset.csv
├── results/
│   └── .stubs/                    # Expected output docs per subdirectory
├── docs/
│   └── AIMS_opcoes_referencia.md  # Complete options reference (artifact details)
└── requirements.txt
```

## Detailed Reference

For a complete reference including experiment composition tables, per-artifact inventory, directory tree, artifact counts, and thesis chapter mapping, see **[docs/AIMS_opcoes_referencia.md](docs/AIMS_opcoes_referencia.md)**.

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

- Federal University of State of Rio de Janeiro (UNIRIO)
- Dataset based on [saraivacode/framework_its_sdn](https://github.com/saraivacode/framework_its_sdn), which uses:
  - [Mininet-WiFi Emulator](https://github.com/intrig-unicamp/mininet-wifi)
  - [Ryu SDN Controller](https://osrg.github.io/ryu/)
  - [SUMO Mobility Simulator](https://sumo.dlr.de/docs/Installing.html)

## Contact

- **Tiago do Vale Saraiva** - [tiago.saraiva@uniriotec.br](mailto:tiago.saraiva@uniriotec.br)
