# AIMS: Adaptive and Intelligent Management of Slicing for Next-Generation ITS Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the **AIMS (Adaptive and Intelligent Management of Slicing)** framework for classifying how network slicing policies impact Intelligent Transportation Systems (ITS) applications. The framework uses machine learning to analyze Quality of Service (QoS) metrics and predict impact levels, enabling dynamic resource allocation in vehicular networks.

## Key Features

- **Multi-model approach**: Implements three complementary classifiers with different strengths
  - Random Forest: Tree-based ensemble with bagging for robust baseline performance
  - TabNet: Deep learning with attention mechanisms optimized for tabular data
  - CatBoost: Gradient boosting with native categorical feature support and advanced regularization
- **Temporal-aware validation**: Uses GroupKFold cross-validation (5 splits) to prevent data leakage in time-series data
- **Feature engineering**: 15+ engineered features from core network metrics
- **Class-balanced training**: Handles imbalanced impact level distribution across all models
- **Automated hyperparameter optimization**: Uses Optuna for efficient parameter tuning
- **Comprehensive evaluation**: Generates confusion matrices, feature importance, and comparative analysis

## Dataset

The framework processes vehicular network QoS data derived from the experimental setup described in T. do Vale Saraiva et al., "An Application-Driven Framework for Intelligent Transportation Systems Using 5G Network Slicing," IEEE Transactions on Intelligent Transportation Systems, vol. 22, no. 8, pp. 5247–5260, Aug. 2021 (based on [saraivacode/framework_its_sdn](https://github.com/saraivacode/framework_its_sdn))

The dataset includes:

- **158 vehicles** traveling on urban roads with QoS data from up to 15 vehicles communicating simultaneously
- **450 seconds** of network measurements
- **4 application classes**: Safety (S), Efficiency (E), Entertainment (E2), Generic (G)
- **Core metrics**: RTT (latency), PDR (packet delivery ratio), Throughput and over 15 engineered features

## Installation

### Prerequisites

```bash
# Python Python3.12.4 or higher
python --version

# Install required packages
pip install -r requirements.txt
```

### Required Libraries

```
catboost==1.2.8
joblib==1.5.1
matplotlib==3.10.3
numpy==2.3.2
optuna==4.4.0
pandas==2.3.1
pytorch_tabnet==4.1.0
scikit_learn==1.7.1
seaborn==0.13.2
tabulate==0.9.0
torch==2.6.0+cu124
```

## Project Structure

```
code/
├── main.py                    # Main training pipeline
├── train_model_catboost.py    # CatBoost training script
├── train_model_rf.py          # Random Forest training script
├── train_model_tabnet.py      # TabNet training script
├── preprocess_dataset.py      # Data preprocessing utilities
├── impact_labeling.py         # Impact level assignment logic
├── save_utils.py              # Artifact saving utilities
├── compare_results.py         # Compare output results script
data/                          # Dataset directory
└── aims_dataset.csv
other/
├── requirements.txt           # Python dependencies
├── README.md                  # This file
```

## Usage

### Quick Start

Train all three models with 15 trials:

```bash
python main.py --compare --csv ../data/aims_dataset.csv --n-trials 15 --n-trials-tabnet 15
```

### Individual Model Training

Train specific models with custom parameters:

```bash
# Custom dataset with increased optimization trials
python main.py --csv ./data/custom_dataset.csv --n-trials 100

# Skip computationally expensive model and generate comparison
python main.py --skip-tabnet --compare

# Development mode - test comparison logic only
python main.py --test-comparison-only

# Targeted training with custom output directory
python main.py --skip-rf --results-dir ./experiments/run_001

# Random Forest
python main.py --compare --csv ../data/aims_dataset.csv --n-trials 15 --skip-catboost --skip-tabnet

# CatBoost
python main.py --compare --csv ../data/aims_dataset.csv --n-trials 15 --skip-rf --skip-tabnet

# TabNet
python main.py --compare --csv ../data/aims_dataset.csv --n-trials-tabnet 15 --skip-catboost --skip-rf
```

### Parameters

**Dataset & Core Configuration:**
- `--csv`: Path to the dataset CSV file (default: ../data/aims_dataset.csv)
- `--random-state`: Random seed for reproducibility (default: 42)
- `--results-dir`: Base directory for storing model results (default: ../results)

**Cross-Validation & Optimization:**
- `--n-splits`: Number of GroupKFold cross-validation splits (default: 5)
- `--n-trials`: Number of Optuna optimization trials for RandomForest and CatBoost (default: 40)
- `--n-trials-tabnet`: Number of Optuna trials for TabNet (default: 40)

**Model Selection:**
- `--skip-rf`: Skip Random Forest training
- `--skip-tabnet`: Skip TabNet training
- `--skip-catboost`: Skip CatBoost training

**Analysis & Testing:**
- `--compare`: Generate comparison report after training completion
- `--test-comparison-only`: Skip training and run only comparison logic

## Impact Classification

The framework uses a weighted-average approach to assign impact levels based on application-specific QoS thresholds. Each metric (RTT, PDR, throughput) is scored 0-3, then combined using application-specific weights.

**Impact Levels:**
- **0 (Adequate)**: Adequate performance, all QoS requirements met
- **1 (Warning)**: Slight degradation, non-critical applications affected
- **2 (Severe)**: Significant degradation, multiple applications impacted
- **3 (Critical)**: Severe degradation, safety applications at risk

**Application Weights:**

| Application | Latency | Loss | Throughput | Priority |
|------------|---------|------|------------|----------|
| Safety (S) | 0.5 | 0.3 | 0.2 | Critical |
| Efficiency (E) | 0.3 | 0.4 | 0.3 | High |
| Entertainment (E2) | 0.2 | 0.3 | 0.5 | Medium |
| Generic (G) | 0.3 | 0.3 | 0.4 | Low |

## Feature Engineering

The framework generates 15+ features from core metrics including temporal features (rolling mean/std with 3-sample window, rate of change), derived metrics (loss ratio, throughput utilization), and categorical data (one-hot encoded application categories).

## Output Artifacts

The framework generates comprehensive outputs for analysis and deployment:

**Model Files**: Trained models saved in pickle format for deployment and inference

**Performance Analysis**: Training results with metrics, confusion matrices (both raw and normalized), and feature importance visualizations

**Optimization Reports**: Hyperparameter optimization history and training curves for model tuning analysis

**Data Exports**: Processed datasets, group classifications, and class weights for reproducibility

**Comparative Analysis**: Model comparison results and consolidated performance metrics across all approaches


## Citation

If you use this code in your research, please cite:

```bibtex
@article{saraiva2025aims,
  title={AIMS: Adaptive and Intelligent Management of Slicing for Next-Generation ITS Networks},
  author={Saraiva, Tiago do Vale},
  journal={},
  year={2025},
  publisher={}
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
