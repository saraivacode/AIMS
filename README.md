# AIMS Framework - ML-based Network Slicing Impact Classification for ITS

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation of the **AIMS (Adaptive and Intelligent Management of Slicing)** framework for classifying the impact of network slicing policies on Intelligent Transportation Systems (ITS) applications. The framework uses machine learning to analyze Quality of Service (QoS) metrics and predict impact levels, enabling dynamic resource allocation in vehicular networks.

### Key Features

- **Multi-model approach**: Implements CatBoost, Random Forest, and TabNet classifiers
- **Temporal-aware validation**: Uses GroupKFold to prevent data leakage in time-series data
- **Comprehensive feature engineering**: 20+ engineered features from core network metrics
- **Class-balanced training**: Handles imbalanced data across impact levels
- **Automated hyperparameter optimization**: Uses Optuna for efficient parameter search

## Dataset

The framework uses a realistic ITS dataset containing:
- **158 vehicles** traveling on urban roads
- **450 seconds** of network measurements
- **4 application classes**: Safety (S), Efficiency (E), Entertainment (E2), Generic (G)
- **Core metrics**: RTT (latency), PDR (packet delivery ratio), Throughput

## Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install required packages
pip install -r requirements.txt
```

### Required Libraries

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
catboost>=1.0.0
pytorch-tabnet>=4.0
optuna>=3.0.0
torch>=1.10.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
```

## Project Structure

```
aims-ml-its-slicing/
├── main.py                    # Main training pipeline
├── train_model_catboost.py    # CatBoost training script
├── train_model_rf.py          # Random Forest training script
├── train_model_tabnet.py      # TabNet training script
├── preprocess_dataset.py      # Data preprocessing utilities
├── impact_labeling.py         # Impact level assignment logic
├── save_utils.py              # Artifact saving utilities
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── d3/                        # Dataset directory
    └── df_completo_com_interpolacao.csv
```

## Usage

### Quick Start

Train all three models with default settings:

```bash
python main.py
```

### Individual Model Training

Train specific models with custom parameters:

```bash
# Random Forest
python train_model_rf.py --csv d3/df_completo_com_interpolacao.csv --n-splits 5 --n-trials 40

# CatBoost
python train_model_catboost.py --csv d3/df_completo_com_interpolacao.csv --n-splits 5 --n-trials 40

# TabNet
python train_model_tabnet.py --csv d3/df_completo_com_interpolacao.csv --n-splits 5 --n-trials 20
```

### Parameters

- `--csv`: Path to the dataset CSV file
- `--n-splits`: Number of GroupKFold cross-validation splits (default: 5)
- `--n-trials`: Number of Optuna optimization trials (default: 40)
- `--random-state`: Random seed for reproducibility (default: 42)

## Impact Labeling Strategy

The framework uses a weighted-average approach to assign impact levels:

1. **Thresholds**: Application-specific QoS thresholds based on industry standards
2. **Scoring**: Each metric (RTT, PDR, throughput) is scored 0-3
3. **Weighting**: Application-specific weights combine scores into final impact

### Impact Levels
- **0 (Low)**: Adequate performance, all QoS requirements met
- **1 (Minor)**: Slight degradation, non-critical applications affected
- **2 (Major)**: Significant degradation, multiple applications impacted
- **3 (Critical)**: Severe degradation, safety applications at risk

### Application Weights

| Application | Latency | Loss | Throughput | Priority |
|------------|---------|------|------------|----------|
| Safety (S) | 0.5 | 0.3 | 0.2 | Critical |
| Efficiency (E) | 0.3 | 0.4 | 0.3 | High |
| Entertainment (E2) | 0.2 | 0.3 | 0.5 | Medium |
| Generic (G) | 0.3 | 0.3 | 0.4 | Low |

## Results

### Model Performance (Holdout Set)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| CatBoost | 94.05% | 94.33% | 94.19% | 94.24% |
| Random Forest | 94.66% | 95.67% | 93.94% | 94.76% |
| **TabNet** | **95.15%** | **94.93%** | **95.85%** | **95.38%** |

### Output Artifacts

Each model training produces:
- `{model}_best.pkl`: Trained model (scikit-learn/PyTorch format)
- `training_results_{model}.json`: Performance metrics and parameters
- `confusion_matrix_{model}.png`: Confusion matrix visualization
- `{model}_optuna_history.html`: Hyperparameter optimization history

## Feature Engineering

The framework generates 20 features from core metrics:

### Temporal Features
- Rolling mean/std (3-sample window)
- Rate of change (delta)

### Derived Metrics
- Loss ratio: (1-PDR)/PDR
- Throughput utilization: throughput/reference_throughput

### Categorical
- Application category (one-hot encoded)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{saraiva2025aims,
  title={Adaptive and Intelligent Management of Slicing for Next-Generation ITS Networks},
  author={Saraiva, Tiago do Vale},
  journal={},
  year={2025},
  publisher={}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset based on SUMO vehicular simulations
- Federal University of State of Rio de Janeiro (UNIRIO)

## Contact

- **Tiago do Vale Saraiva** - [tiago.saraiva@uniriotec.br](mailto:tiago.saraiva@uniriotec.br)

Project Link: [https://github.com/saraivacode/AIMS](https://github.com/saraivacode/AIMS)
