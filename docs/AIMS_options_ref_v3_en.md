---
type: organization
tags: [AIMS, deep-learning, federated-learning, security, simulation]
---

# AIMS Framework — Complete Options Reference

> Auto-generated from source code (`code/main.py`, `code/federated/fl_main.py`, `code/federated/fl_config.py`).
> Last updated: 2026-03-22 (v3 — full source code audit including Ch. 6/7)

---

## Changelog

### v3 (current)

| Section | Change | Detail |
|---|---|---|
| §3.3 | **Artifact tables rewritten per model** | Generic `{suffix}` template replaced by exact filenames from each `train_model_*.py`. Inconsistent suffix conventions now documented explicitly (e.g., CatBoost `feature_importance` singular vs RF/TabNet `feature_importances` plural). |
| §3.3 | **`model_comparison.py` → `compare_results.py`** | Module name corrected to match actual source file and import in `main.py`. |
| §3.3 | **RF: no training curves** | Documented that RF does not generate `*_training_curves.png` (non-iterative model). |
| §4.4 | **FL JSON naming order corrected** | `{Model}_{Dist}_{Strat}.json` → `{Model}_{Strategy}_{Dist}.json` to match `fl_server.py` `config_name`. |
| §6 | **Directory tree updated** | Per-model filenames now match actual code output. |

### v2

| Section | Change | Detail |
|---|---|---|
| §5 Security | **Experiment counts corrected** | Code iterates over 2 strategies (FedAvg, FedProx) × 3 defenses (baseline, Krum, TrimmedMean) = **6 combos**, not 3. Phase 1a: 9→**18**, Phase 1b: 27→**54**, Sensitivity: 36→**72**, Total Ch.8: 72→**144** |
| §5.1 | `--security-only` flag documented | Was missing from v1 |
| §5.2 | Phase detail tables updated | Now show strategy × defense matrix explicitly |
| §5.5 | Artifact counts and naming updated | JSON counts doubled; naming convention includes `{Strategy}` and `_def-{Defense}` |
| §6 | Directory tree updated | File naming pattern corrected to match `fl_server.py` `config_name` |
| §7 | Numeric summary corrected | All Ch.8 totals updated |
| §8 | Chapter mapping totals corrected | Ch.8: 72→144, grand total updated |

---

## 1. Default Values (fl_config.py)

| Parameter | Default | Description |
|---|---|---|
| `NUM_CLIENTS` | 3 | Number of simulated FL clients (RSUs) |
| `NUM_ROUNDS` | 30 | FL communication rounds |
| `LOCAL_EPOCHS` | 3 | Local training epochs per round |
| `BATCH_SIZE` | 32 | Batch size |
| `STRATEGIES` | FedAvg, FedProx | Aggregation strategies |
| `FEDPROX_MU` | 0.1 | FedProx proximity parameter |
| `DISTRIBUTIONS` | IID, NonIID | Data distribution modes |
| `MODEL_TYPES` | DNN, LSTM, GRU | Neural network architectures |
| `CENTRALIZED_EPOCHS` | 50 | Epochs for centralized baselines (DNN/LSTM/GRU) |
| `EARLY_STOPPING_PATIENCE` | 10 | Early stopping patience |
| `RANDOM_STATE` | 42 | Reproducibility seed |

---

## 2. Experiment ID and Results Directory

Every run creates an isolated output directory under `results/`:

```
results/<experiment-id>/
├── random_forest/
├── tabnet/
├── catboost/
├── model_comparison.{png,csv}
└── federated/
    ├── security_phase1a/
    ├── security_phase1b/
    └── security_sensitivity_epochs/
```

| Option | Default | Description |
|---|---|---|
| `--experiment-id` | UTC timestamp (`20260321T143052Z`) | String identifier for the experiment run. All outputs are saved under `results-dir/<experiment-id>/`. |
| `--results-dir` | `../results` | Base directory for results. Combined with `--experiment-id` to form the final path. |

**Examples:**

```bash
# Auto-generated timestamp ID (default)
python main.py --federated-only
# → results/20260321T143052Z/federated/...

# Custom experiment ID
python main.py --federated-only --experiment-id "baseline-v1"
# → results/baseline-v1/federated/...

# Re-running with different params, separate ID
python main.py --federated-only --experiment-id "fedprox-mu02" --fl-rounds 50
# → results/fedprox-mu02/federated/...
```

> Results from different runs never overwrite each other. Each `--experiment-id` creates a completely separate directory tree.

---

## 3. CLI Options — Centralized Models (Ch. 6)

### 3.1 Main Flags

| Flag | Effective Models | # Experiments | Purpose |
|---|---|---|---|
| *(default, no skip)* | **RF + TabNet + CatBoost** | 3 HPO pipelines | Which centralized model is best for vehicular impact classification? |
| `--skip-tabnet --skip-catboost` | **Random Forest** | 1 HPO pipeline | Tree-based ensemble baseline |
| `--skip-rf --skip-catboost` | **TabNet** | 1 HPO pipeline | Does attention-based DL improve over RF? |
| `--skip-rf --skip-tabnet` | **CatBoost** | 1 HPO pipeline | Is gradient boosting with categoricals better? |
| `--compare` | *(post-processing)* | 0 (generates report after training) | Direct comparison across all trained models |
| `--test-comparison-only` | *(post-processing)* | 0 (skips training) | Re-generate comparison without re-training |

### 3.2 Configurable Options

| Option | Default | Applies to | Description |
|---|---|---|---|
| `--n-trials` | 40 | RF, CatBoost | Number of Optuna trials |
| `--n-trials-tabnet` | 40 | TabNet | Number of Optuna trials (separate, more expensive) |
| `--n-splits` | 5 | RF, TabNet, CatBoost | GroupKFold CV folds |
| `--random-state` | 42 | All | Reproducibility seed |
| `--csv` | `../data/aims_dataset.csv` | All | Dataset path |

### 3.3 Artifacts per Centralized Model

All three models share a common set of artifacts generated by `save_utils.save_model_results()`:

| Artifact | Format | Description |
|---|---|---|
| `X_{model}.csv` | CSV | Feature matrix |
| `y_{model}.csv` / `.npy` | CSV + NPY | Labels (text + binary) |
| `groups_{model}.csv` / `.npy` | CSV + NPY | Group IDs for temporal CV |
| `class_weight_{model}.json` | JSON | Class weights for balancing |

Where `{model}` is `random_forest`, `tabnet`, or `catboost`.

Each model then generates additional artifacts via its training script and `ResultsManager`. **Note:** the suffix convention is inconsistent across models in the current codebase — the exact filenames per model are listed below.

#### Random Forest → `random_forest/`

| Filename | Format | Description |
|---|---|---|
| `random_forest_model.pkl` | Pickle/Joblib | Serialized trained model |
| `training_results_random_forest.json` | JSON | Full results: hyperparams, CV metrics, holdout metrics, HPO stats, system info |
| `feature_importances_rf.csv` | CSV | Feature importance (sorted) |
| `confusion_matrix_rf_raw.png` | PNG | Confusion matrix (absolute values) |
| `confusion_matrix_rf_normalized.png` | PNG | Confusion matrix (normalized) |
| `optuna_history_rf.html` | HTML | Interactive Optuna optimization history (Plotly) |
| `feature_importance_rf.png` | PNG | Top 20 features barplot |
| `feature_importance_cumulative_rf.png` | PNG | Cumulative importance (90%, 95%, 99% thresholds) |

> RF does not generate training curves (non-iterative model — independent trees built in parallel).

#### TabNet → `tabnet/`

| Filename | Format | Description |
|---|---|---|
| `tabnet_model.pkl` | Pickle | Serialized trained model |
| `training_results_tabnet.json` | JSON | Full results |
| `feature_importances_tabnet.csv` | CSV | Feature importance (sorted) |
| `confusion_matrix_tb_raw.png` | PNG | Confusion matrix (absolute values) |
| `confusion_matrix_tb_normalized.png` | PNG | Confusion matrix (normalized) |
| `optuna_history_tb.html` | HTML | Optuna optimization history |
| `feature_importance_tabnet.png` | PNG | Top 20 features barplot |
| `feature_importance_cumulative_tabnet.png` | PNG | Cumulative importance |
| `tabnet_training_curves.png` | PNG | Loss + accuracy/F1 curves over training |

#### CatBoost → `catboost/`

| Filename | Format | Description |
|---|---|---|
| `catboost_model.pkl` | Pickle | Serialized trained model |
| `training_results_catboost.json` | JSON | Full results |
| `feature_importance_catboost.csv` | CSV | Feature importance (sorted) — **note: singular** (`feature_importance`, not `feature_importances`) |
| `confusion_matrix_cb_raw.png` | PNG | Confusion matrix (absolute values) |
| `confusion_matrix_cb_normalized.png` | PNG | Confusion matrix (normalized) |
| `optuna_history_catboost.html` | HTML | Optuna optimization history |
| `feature_importance_catboost.png` | PNG | Top 20 features barplot |
| `feature_importance_cumulative_catboost.png` | PNG | Cumulative importance |
| `catboost_training_curves.png` | PNG | Loss + accuracy/F1 curves over training |

#### Suffix Summary

The suffix conventions are inconsistent across models. For reference:

| Artifact type | RF | TabNet | CatBoost |
|---|---|---|---|
| Model PKL | `random_forest` | `tabnet` | `catboost` |
| Results JSON | `random_forest` | `tabnet` | `catboost` |
| Confusion matrix PNG | `rf` | `tb` | `cb` |
| Optuna history HTML | `rf` | `tb` | `catboost` |
| Feature importance PNG | `rf` | `tabnet` | `catboost` |
| Feature importance CSV | `rf` (plural) | `tabnet` (plural) | `catboost` (**singular**) |

**`--compare` artifacts** (generated by `compare_results.py` → `generate_report()`, saved at experiment root):

| Artifact | Format | Description |
|---|---|---|
| `model_comparison.png` | PNG | 4 panels: Accuracy, CV vs Test F1, Per-class F1 heatmap, Training time |
| `model_comparison.csv` | CSV | Comparative table: model, metrics, timing, hyperparameters |

---

## 4. CLI Options — Federated Learning (Ch. 7)

### 4.1 Main Flags

| Flag | What runs | # Experiments | Purpose |
|---|---|---|---|
| `--federated` | **DNN, LSTM, GRU** federated (FedAvg/FedProx × IID/NonIID) + **DNN, LSTM, GRU** centralized baselines + **RF, TabNet, CatBoost** centralized | **12 FL** + **3 NN baselines** + **3 ML centralized** = **18 total** | Does FL preserve performance vs centralized? Does FedProx help on NonIID? |
| `--federated-only` | **DNN, LSTM, GRU** federated (FedAvg/FedProx × IID/NonIID) + **DNN, LSTM, GRU** centralized baselines. **No RF/TabNet/CatBoost.** | **12 FL** + **3 NN baselines** = **15 total** | Same as above, without traditional ML models |

> **`--skip-centralized`**: Removes the centralized baselines (DNN/LSTM/GRU) from either option. Only the 12 federated experiments remain.

### 4.2 Composition: `--federated` vs `--federated-only`

| Component | `--federated` | `--federated-only` | `+ --skip-centralized` |
|---|---|---|---|
| DNN/LSTM/GRU federated (3×2×2 = 12 exp) | yes | yes | yes |
| DNN/LSTM/GRU centralized baseline (3 exp) | yes | yes | no |
| RF + TabNet + CatBoost centralized (3 HPO pipelines) | yes | no | no |

### 4.3 Configurable Options

| Option | Default | Description |
|---|---|---|
| `--fl-rounds` | 30 | FL communication rounds |
| `--fl-local-epochs` | 3 | Local training epochs per round |
| `--fl-clients` | 3 | Number of simulated clients (RSUs) |
| `--fl-strategies` | `FedAvg FedProx` | Aggregation strategies (accepts multiple) |
| `--fl-distributions` | `IID NonIID` | Distribution modes (accepts multiple) |
| `--fl-models` | `DNN LSTM GRU` | Neural architectures (accepts multiple) |
| `--skip-centralized` | `False` | Skip centralized DNN/LSTM/GRU baselines |

### 4.4 Artifacts (Federated Learning)

All saved under `<experiment-id>/federated/`. Generated by `FLResultsManager` (`fl_results.py`) and `fl_visualizations.py`.

#### Per-experiment files

| Artifact | Format | Generated by | Description |
|---|---|---|---|
| `{Model}_{Strategy}_{Dist}.json` | JSON | `FLResultsManager.save_experiment()` | Full result: model_info, fl_config, per-round metrics, convergence, history |
| `{Model}_centralized.json` | JSON | `FLResultsManager.save_experiment()` | Centralized baseline (50 epochs, early stopping) |

> **Examples:** `DNN_FedAvg_IID.json`, `LSTM_FedProx_NonIID.json`, `GRU_centralized.json`

#### Aggregated outputs

| Artifact | Format | Generated by | Description |
|---|---|---|---|
| `summary_table.csv` | CSV | `FLResultsManager.save_summary_table()` | Accuracy, Precision, Recall, F1-Macro, @85%, @90%, Stability, Time |
| `summary_table.tex` | TEX | `FLResultsManager.save_summary_table()` | Same table in LaTeX |
| `centralized_results.json` | JSON | `FLResultsManager` | All centralized baselines in one JSON |
| `client_distribution_IID.png` | PNG | `fl_visualizations.py` | Class distribution across IID clients |
| `client_distribution_NonIID.png` | PNG | `fl_visualizations.py` | Class distribution across NonIID clients |
| `convergence_comparison.png` | PNG | `fl_visualizations.py` | 3 subplots (1/model): accuracy curves per strategy/dist, 85%/90% refs |
| `strategy_comparison.png` | PNG | `fl_visualizations.py` | FedAvg vs FedProx on NonIID |
| `fl_vs_centralized.png` | PNG | `fl_visualizations.py` | F1-Macro: best FL config vs centralized baseline per model |

---

## 5. CLI Options — FL Security (Ch. 8)

### 5.1 Main Flags

| Flag | Phases executed | # Exp | Purpose |
|---|---|---|---|
| `--security` | **Phase 1a** (label-flip) + **Phase 1b** (label-flip + gradient scaling 2×/5×/10×) | **18 + 54 = 72** | Does label-flip degrade FL? Does gradient scaling amplify it? Do Krum/TrimmedMean protect? How does FedProx compare to FedAvg under attack? |
| `--security --skip-phase1a` | Only **Phase 1b** | **54** | Focus on gradient scaling only |
| `--security --skip-phase1b` | Only **Phase 1a** | **18** | Focus on label-flip only |
| `--security-only` | Same as `--security` but skips RF/TabNet/CatBoost | **72** | Convenience flag: implies `--security --skip-rf --skip-tabnet --skip-catboost` |
| `--security-sensitivity-epochs` | Epoch sensitivity (1, 3, 5, 10) under label-flip | **72** | Do more local epochs worsen convergence under attack? What is the optimal epoch count? |

> **Combined command for all Ch. 8 experiments:**
> ```bash
> python main.py --security --security-sensitivity-epochs \
>     --skip-rf --skip-tabnet --skip-catboost \
>     --experiment-id "ch8-security" --csv ../data/aims_dataset.csv
> ```
> This runs Phase 1a (18) + Phase 1b (54) + Sensitivity (72) = **144 experiments total**.

### 5.2 Phase Details

All security phases iterate over a fixed list of **6 strategy×defense combinations** defined in `fl_main.py`:

```python
strategies_and_defenses = [
    ("FedAvg",  "FedAvg"),       # FedAvg baseline (no robust defense)
    ("FedProx", "FedProx"),      # FedProx baseline (no robust defense)
    ("FedAvg",  "Krum"),         # FedAvg + Krum defense
    ("FedProx", "Krum"),         # FedProx + Krum defense
    ("FedAvg",  "TrimmedMean"),  # FedAvg + TrimmedMean defense
    ("FedProx", "TrimmedMean"),  # FedProx + TrimmedMean defense
]
```

This means each security phase tests **2 aggregation strategies × 3 defense modes = 6 combinations** per model (and per scale/epoch variant), not 3 as might be assumed from listing only the defenses.

#### Phase 1a — Pure Label-Flip

| Parameter | Value |
|---|---|
| Attack | `label_flip` (Critical → Adequate) |
| Malicious clients | 1 of 3 (client 0) |
| Gradient scaling | None (1×) |
| Strategies | FedAvg, FedProx |
| Defenses | Baseline (no defense), Krum, TrimmedMean |
| Models | DNN, LSTM, GRU |
| Distribution | NonIID |
| Total | 3 models × 2 strategies × 3 defenses = **18 experiments** |

#### Phase 1b — Label-Flip + Gradient Scaling

| Parameter | Value |
|---|---|
| Attack | `label_flip+gradient_scaling` |
| Malicious clients | 1 of 3 (client 0) |
| Gradient scaling | 2×, 5×, 10× |
| Strategies | FedAvg, FedProx |
| Defenses | Baseline (no defense), Krum, TrimmedMean |
| Models | DNN, LSTM, GRU |
| Distribution | NonIID |
| Total | 3 models × 3 scales × 2 strategies × 3 defenses = **54 experiments** |

#### Sensitivity Epochs

| Parameter | Value |
|---|---|
| Attack | `label_flip` (Critical → Adequate) |
| Local epochs | **1, 3, 5, 10** (varies) |
| Strategies | FedAvg, FedProx |
| Defenses | Baseline (no defense), Krum, TrimmedMean |
| Models | DNN, LSTM, GRU |
| Distribution | NonIID |
| Total | 3 models × 4 epochs × 2 strategies × 3 defenses = **72 experiments** |

### 5.3 Configurable Options

| Option | Default | Applies to | Note |
|---|---|---|---|
| `--fl-rounds` | 30 | All phases | Communication rounds |
| `--fl-local-epochs` | 3 | Phase 1a, Phase 1b | **Ignored** in sensitivity (varies internally: 1, 3, 5, 10) |
| `--fl-models` | `DNN LSTM GRU` | All phases | Neural architectures |
| `--random-state` | 42 | All phases | Seed |
| `--csv` | `../data/aims_dataset.csv` | All phases | Dataset |

### 5.4 Fixed Parameters (not configurable via CLI)

| Parameter | Value | Description |
|---|---|---|
| Distribution | NonIID | All security phases use NonIID only |
| # clients | 3 | Fixed at 3 RSUs |
| Malicious clients | `[0]` (1 of 3) | Always client 0 |
| Batch size | 32 | Hardcoded |
| FedProx μ | 0.1 | Hardcoded in `fl_config.py` |

### 5.5 Artifacts (Security)

Generated by `FLResultsManager` (`fl_results.py`) and `fl_visualizations.py`.

#### File Naming Convention

Security experiment JSON files follow the naming pattern generated by `fl_server.py`:

```
{Model}_{Strategy}_{Distribution}_attack-{attack_type}[_x{scale}]_def-{Defense}[_ep{epochs}].json
```

Where:
- `{Model}`: DNN, LSTM, GRU
- `{Strategy}`: FedAvg, FedProx
- `{Distribution}`: NonIID (always)
- `{attack_type}`: `label_flip` or `label_flip+gradient_scaling`
- `_x{scale}`: only present when gradient_scale > 1.0 (e.g., `_x2`, `_x5`, `_x10`)
- `_def-{Defense}`: FedAvg, FedProx (baseline), Krum, TrimmedMean
- `_ep{epochs}`: only present when local_epochs ≠ default (3)

#### Phase 1a → `<experiment-id>/federated/security_phase1a/`

| Artifact | Format | Count | Description |
|---|---|---|---|
| `{Model}_{Strategy}_NonIID_attack-label_flip_def-{Defense}.json` | JSON | **18** | Result: attack config, C2A rate, per-round metrics |
| `phase1a_summary.csv` | CSV | 1 | Model, Strategy, Defense, Attack, Accuracy, F1-Macro, C2A Rate, @85%, @90%, Time |
| `phase1a_summary.tex` | TEX | 1 | Same in LaTeX |
| `phase1a_c2a_comparison.png` | PNG | 1 | Barplot: C2A rate by model and defense |
| `phase1a_accuracy_vs_c2a.png` | PNG | 1 | Scatter: accuracy vs C2A, colored by defense |
| `phase1a_convergence.png` | PNG | 1 | 3 subplots: accuracy convergence per defense, 1/model |
| `phase1a_c2a_convergence.png` | PNG | 1 | 3 subplots: C2A rate over rounds |

> **Example JSONs:**
> `DNN_FedAvg_NonIID_attack-label_flip_def-FedAvg.json`,
> `DNN_FedProx_NonIID_attack-label_flip_def-Krum.json`,
> `LSTM_FedAvg_NonIID_attack-label_flip_def-TrimmedMean.json`

#### Phase 1b → `<experiment-id>/federated/security_phase1b/`

| Artifact | Format | Count | Description |
|---|---|---|---|
| `{Model}_{Strategy}_NonIID_attack-label_flip+gradient_scaling_x{Scale}_def-{Defense}.json` | JSON | **54** | Result with gradient_scale, C2A rate |
| `phase1b_summary.csv` | CSV | 1 | Includes Scale column |
| `phase1b_summary.tex` | TEX | 1 | LaTeX |
| `phase1b_c2a_comparison.png` | PNG | 1 | C2A by model and defense |
| `phase1b_accuracy_vs_c2a.png` | PNG | 1 | Accuracy vs C2A |
| `phase1b_convergence.png` | PNG | 1 | Convergence with scale labels |
| `phase1b_c2a_convergence.png` | PNG | 1 | C2A convergence |

> **Example JSONs:**
> `DNN_FedAvg_NonIID_attack-label_flip+gradient_scaling_x2_def-FedAvg.json`,
> `LSTM_FedProx_NonIID_attack-label_flip+gradient_scaling_x5_def-Krum.json`,
> `GRU_FedAvg_NonIID_attack-label_flip+gradient_scaling_x10_def-TrimmedMean.json`

#### Sensitivity Epochs → `<experiment-id>/federated/security_sensitivity_epochs/`

| Artifact | Format | Count | Description |
|---|---|---|---|
| `{Model}_{Strategy}_NonIID_attack-label_flip_def-{Defense}[_ep{N}].json` | JSON | **72** | Result with local_epochs (note: `_ep3` omitted when epochs=3=default) |
| `sensitivity_epochs_summary.csv` | CSV | 1 | Sensitivity table |
| `sensitivity_epochs_summary.tex` | TEX | 1 | LaTeX |
| `security_sensitivity_epochs.png` | PNG | 1 | 2×3 subplots: accuracy + C2A vs epochs, lines per defense |
| `sensitivity_c2a_comparison.png` | PNG | 1 | C2A comparison |

> **Example JSONs:**
> `DNN_FedAvg_NonIID_attack-label_flip_def-FedAvg_ep1.json`,
> `DNN_FedProx_NonIID_attack-label_flip_def-FedProx.json` (ep3 = default, no suffix),
> `LSTM_FedAvg_NonIID_attack-label_flip_def-Krum_ep5.json`,
> `GRU_FedProx_NonIID_attack-label_flip_def-TrimmedMean_ep10.json`

---

## 6. Results Directory Structure

```
results/
├── .stubs/                                    ← checked into git (expected output docs)
│   ├── EXPECTED_OUTPUTS.md
│   ├── random_forest/EXPECTED_OUTPUTS.md
│   ├── tabnet/EXPECTED_OUTPUTS.md
│   ├── catboost/EXPECTED_OUTPUTS.md
│   └── federated/
│       ├── EXPECTED_OUTPUTS.md
│       ├── security_phase1a/EXPECTED_OUTPUTS.md
│       ├── security_phase1b/EXPECTED_OUTPUTS.md
│       └── security_sensitivity_epochs/EXPECTED_OUTPUTS.md
│
├── baseline-v1/                               ← experiment run (gitignored)
│   ├── random_forest/                         ← Ch. 6: RF centralized
│   │   ├── X_random_forest.csv
│   │   ├── y_random_forest.{csv,npy}
│   │   ├── groups_random_forest.{csv,npy}
│   │   ├── class_weight_random_forest.json
│   │   ├── random_forest_model.pkl
│   │   ├── feature_importances_rf.csv
│   │   ├── training_results_random_forest.json
│   │   ├── confusion_matrix_rf_{raw,normalized}.png
│   │   ├── optuna_history_rf.html
│   │   ├── feature_importance_rf.png
│   │   └── feature_importance_cumulative_rf.png
│   │
│   ├── tabnet/                                ← Ch. 6: TabNet centralized
│   │   ├── X_tabnet.csv
│   │   ├── y_tabnet.{csv,npy}
│   │   ├── groups_tabnet.{csv,npy}
│   │   ├── class_weight_tabnet.json
│   │   ├── tabnet_model.pkl
│   │   ├── feature_importances_tabnet.csv
│   │   ├── training_results_tabnet.json
│   │   ├── confusion_matrix_tb_{raw,normalized}.png   ← note: suffix "tb"
│   │   ├── optuna_history_tb.html                     ← note: suffix "tb"
│   │   ├── feature_importance_tabnet.png              ← note: suffix "tabnet"
│   │   ├── feature_importance_cumulative_tabnet.png
│   │   └── tabnet_training_curves.png
│   │
│   ├── catboost/                              ← Ch. 6: CatBoost centralized
│   │   ├── X_catboost.csv
│   │   ├── y_catboost.{csv,npy}
│   │   ├── groups_catboost.{csv,npy}
│   │   ├── class_weight_catboost.json
│   │   ├── catboost_model.pkl
│   │   ├── feature_importance_catboost.csv            ← note: singular (no "s")
│   │   ├── training_results_catboost.json
│   │   ├── confusion_matrix_cb_{raw,normalized}.png   ← note: suffix "cb"
│   │   ├── optuna_history_catboost.html               ← note: suffix "catboost"
│   │   ├── feature_importance_catboost.png
│   │   ├── feature_importance_cumulative_catboost.png
│   │   └── catboost_training_curves.png
│   │
│   ├── model_comparison.png                   ← Ch. 6: --compare (compare_results.py)
│   ├── model_comparison.csv
│   │
│   └── federated/                             ← Ch. 7 + Ch. 8
│       ├── {Model}_{Strategy}_{Dist}.json     (12 FL experiments)
│       ├── {Model}_centralized.json           (3 baselines)
│       ├── centralized_results.json
│       ├── summary_table.{csv,tex}
│       ├── client_distribution_{IID,NonIID}.png
│       ├── convergence_comparison.png
│       ├── strategy_comparison.png
│       ├── fl_vs_centralized.png
│       │
│       ├── security_phase1a/                  ← Ch. 8: label-flip
│       │   ├── {Model}_{Strategy}_NonIID_attack-label_flip_def-{Defense}.json  (18 files)
│       │   ├── phase1a_summary.{csv,tex}
│       │   └── phase1a_*.png                  (4 plots)
│       │
│       ├── security_phase1b/                  ← Ch. 8: label-flip + gradient scaling
│       │   ├── {Model}_{Strategy}_NonIID_attack-label_flip+gradient_scaling_x{Scale}_def-{Defense}.json  (54 files)
│       │   ├── phase1b_summary.{csv,tex}
│       │   └── phase1b_*.png                  (4 plots)
│       │
│       └── security_sensitivity_epochs/       ← Ch. 8: epoch sensitivity
│           ├── {Model}_{Strategy}_NonIID_attack-label_flip_def-{Defense}[_ep{N}].json  (72 files)
│           ├── sensitivity_epochs_summary.{csv,tex}
│           └── security_sensitivity_*.png     (2 plots)
│
└── 20260321T143052Z/                          ← another run (auto-generated ID)
    └── ...                                    ← same structure
```

### Artifact Count per Experiment Run

| Directory | JSON | PNG | CSV | TEX | HTML | PKL | NPY | Total |
|---|---|---|---|---|---|---|---|---|
| `random_forest/` | 2 | 4 | 4 | — | 1 | 1 | 2 | **14** |
| `tabnet/` | 2 | 5 | 4 | — | 1 | 1 | 2 | **15** |
| `catboost/` | 2 | 5 | 4 | — | 1 | 1 | 2 | **15** |
| *(root)* | — | 1 | 1 | — | — | — | — | **2** |
| `federated/` | 16 | 5 | 1 | 1 | — | — | — | **23** |
| `federated/security_phase1a/` | 18 | 4 | 1 | 1 | — | — | — | **24** |
| `federated/security_phase1b/` | 54 | 4 | 1 | 1 | — | — | — | **60** |
| `federated/security_sensitivity_epochs/` | 72 | 2 | 1 | 1 | — | — | — | **76** |
| **TOTAL** | **166** | **30** | **17** | **4** | **3** | **3** | **6** | **~229** |

---

## 7. Numeric Summary

| Flag | Exp. | Models | Distributions | Strategies | Defenses | Scales | Epochs |
|---|---|---|---|---|---|---|---|
| *(default)* | 3 HPO pipelines | RF, TabNet, CatBoost | — (centralized) | — | — | — | — |
| `--compare` | 0 (report) | RF, TabNet, CatBoost | — | — | — | — | — |
| `--federated` | 18 | DNN/LSTM/GRU (FL+cent.) + RF/TabNet/CatBoost (cent.) | IID, NonIID | FedAvg, FedProx | — | — | 3 |
| `--federated-only` | 15 | DNN/LSTM/GRU (FL + centralized baseline) | IID, NonIID | FedAvg, FedProx | — | — | 3 |
| `--security` | **72** | DNN, LSTM, GRU | NonIID | FedAvg, FedProx | Baseline, Krum, TrimmedMean | 1× + 2×/5×/10× | 3 |
| `--security --skip-phase1b` | **18** | DNN, LSTM, GRU | NonIID | FedAvg, FedProx | Baseline, Krum, TrimmedMean | 1× | 3 |
| `--security --skip-phase1a` | **54** | DNN, LSTM, GRU | NonIID | FedAvg, FedProx | Baseline, Krum, TrimmedMean | 2×, 5×, 10× | 3 |
| `--security-sensitivity-epochs` | **72** | DNN, LSTM, GRU | NonIID | FedAvg, FedProx | Baseline, Krum, TrimmedMean | 1× | 1, 3, 5, 10 |

---

## 8. Thesis Chapter Mapping

### Ch. 6 — AIMS: Centralized Classification (ISCC 2026)

**Overleaf file:** `0140_cap6_aims.tex`

Four HPO configurations were used to study convergence behavior and trial budget sensitivity:

```bash
# Exploratory (quick, 5 trials each)
python main.py --compare --experiment-id "ch6-5t" \
    --csv ../data/aims_dataset.csv --n-trials 5 --n-trials-tabnet 5

# Intermediate (15 trials uniform)
python main.py --compare --experiment-id "ch6-15t" \
    --csv ../data/aims_dataset.csv --n-trials 15 --n-trials-tabnet 15

# ★ PRIMARY (used for the main analysis)
python main.py --compare --experiment-id "ch6-iscc" \
    --csv ../data/aims_dataset.csv --n-trials 15 --n-trials-tabnet 40

# Extended TabNet budget
python main.py --compare --experiment-id "ch6-50t" \
    --csv ../data/aims_dataset.csv --n-trials 15 --n-trials-tabnet 50
```

| Run | `--n-trials` (RF/CB) | `--n-trials-tabnet` | Purpose |
|---|---|---|---|
| 5 / 5 | 5 | 5 | Quick exploration, verify pipeline |
| 15 / 15 | 15 | 15 | Uniform budget, convergence baseline |
| **15 / 40** | **15** | **40** | **Primary analysis** — RF/CB converge early, TabNet benefits from more trials |
| 15 / 50 | 15 | 50 | Check if TabNet improves beyond 40 trials (diminishing returns) |

**Models:** RF, TabNet, CatBoost
**Research question:** Which centralized model is best for vehicular impact prediction? How many HPO trials are needed for convergence?
**Key artifacts:** `training_results_*.json`, `confusion_matrix_*.png`, `model_comparison.{png,csv}`, `optuna_history_*.html`

---

### Ch. 7 — Federated Classification (VTC 2026)

**Overleaf file:** `0145_cap7_fl.tex`

```bash
python main.py --federated-only --experiment-id "ch7-vtc" \
    --csv ../data/aims_dataset.csv
```

**Models:** DNN, LSTM, GRU (federated: FedAvg/FedProx × IID/NonIID) + DNN, LSTM, GRU (centralized baseline)
**Research question:** Does FL preserve performance vs centralized? Does FedProx help on NonIID?
**Key artifacts:** `summary_table.{csv,tex}`, `convergence_comparison.png`, `fl_vs_centralized.png`, `strategy_comparison.png`

---

### Ch. 8 — ITS Security

**Overleaf file:** `0150_cap8_seguranca.tex`

```bash
# All security (phase 1a + 1b + sensitivity) — 144 experiments total
python main.py --security --security-sensitivity-epochs \
    --skip-rf --skip-tabnet --skip-catboost \
    --experiment-id "ch8-security" \
    --csv ../data/aims_dataset.csv

# Phase 1a only (label-flip) — 18 experiments
python main.py --security --skip-phase1b \
    --skip-rf --skip-tabnet --skip-catboost \
    --experiment-id "ch8-phase1a" \
    --csv ../data/aims_dataset.csv

# Phase 1b only (gradient scaling) — 54 experiments
python main.py --security --skip-phase1a \
    --skip-rf --skip-tabnet --skip-catboost \
    --experiment-id "ch8-phase1b" \
    --csv ../data/aims_dataset.csv

# Epoch sensitivity only — 72 experiments
python main.py --security-sensitivity-epochs \
    --skip-rf --skip-tabnet --skip-catboost \
    --experiment-id "ch8-sensitivity" \
    --csv ../data/aims_dataset.csv
```

**Models:** DNN, LSTM, GRU
**Research question:** How do label-flip and gradient scaling attacks affect vehicular FL? Are Krum and TrimmedMean effective defenses? Does FedProx offer additional resilience under attack? What is the impact of local epochs on robustness?
**Key artifacts:** `phase1a_summary.csv`, `phase1b_summary.csv`, `*_c2a_comparison.png`, `*_convergence.png`, `security_sensitivity_epochs.png`

---

### Summary: Chapter → Flags → Artifacts

| Ch. | Overleaf file | AIMS flags | Models | Exp. | Key artifacts |
|---|---|---|---|---|---|
| **6** | `0140_cap6_aims.tex` | `--compare` (4 trial configs) | RF, TabNet, CatBoost | 3 HPO × 4 runs | `model_comparison.{png,csv}`, `training_results_*.json`, `optuna_history_*.html` |
| **7** | `0145_cap7_fl.tex` | `--federated-only` | DNN, LSTM, GRU (FL + cent.) | 15 | `summary_table.csv`, `convergence_comparison.png`, `fl_vs_centralized.png` |
| **8** | `0150_cap8_seguranca.tex` | `--security`, `--security-sensitivity-epochs` | DNN, LSTM, GRU | **144** | `phase1a_summary.csv`, `phase1b_summary.csv`, `*_c2a_*.png`, `sensitivity_epochs.png` |
| | | **TOTAL** | | **~159 + 3 HPO** | **~229 artifacts per run** |


```bash
(aims) root@0e82ee2e1157:/workspace/AIMS/code# cat /workspace/AIMS/requirements_l40s_working.txt
absl-py==2.4.0
alembic==1.18.4
astunparse==1.6.3
catboost==1.2.8
certifi==2026.2.25
cffi==2.0.0
charset-normalizer==3.4.6
click==8.3.1
colorlog==6.10.1
contourpy==1.3.3
cryptography==46.0.5
cycler==0.12.1
filelock==3.20.0
flatbuffers==25.12.19
flwr==1.27.0
fonttools==4.62.1
fsspec==2025.12.0
gast==0.7.0
google-pasta==0.2.0
graphviz==0.21
greenlet==3.3.2
grpcio==1.78.0
grpcio-health-checking==1.70.0
h5py==3.16.0
idna==3.11
iterators==0.0.2
Jinja2==3.1.6
joblib==1.5.1
keras==3.13.2
kiwisolver==1.5.0
libclang==18.1.1
Mako==1.3.10
Markdown==3.10.2
markdown-it-py==4.0.0
MarkupSafe==3.0.2
matplotlib==3.10.8
mdurl==0.1.2
ml-dtypes==0.4.1
mpmath==1.3.0
namex==0.1.0
networkx==3.6.1
numpy==2.0.2
nvidia-cublas-cu12==12.4.2.65
nvidia-cuda-cupti-cu12==12.4.99
nvidia-cuda-nvrtc-cu12==12.4.99
nvidia-cuda-runtime-cu12==12.4.99
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.0.44
nvidia-curand-cu12==10.3.5.119
nvidia-cusolver-cu12==11.6.0.99
nvidia-cusparse-cu12==12.3.0.142
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.4.99
nvidia-nvtx-cu12==12.4.99
opt_einsum==3.4.0
optree==0.19.0
optuna==4.4.0
packaging==26.0
pandas==3.0.1
pathspec==0.12.1
pillow==12.0.0
plotly==6.1.2
protobuf==5.29.6
pycparser==3.0
pycryptodome==3.23.0
Pygments==2.19.2
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pytorch-tabnet==4.1.0
PyYAML==6.0.3
requests==2.32.5
rich==13.9.4
scikit-learn==1.7.0
scipy==1.17.1
seaborn==0.12.2
shellingham==1.5.4
six==1.17.0
SQLAlchemy==2.0.48
sympy==1.14.0
tabulate==0.9.0
tensorboard==2.18.0
tensorboard-data-server==0.7.2
tensorflow==2.18.0
tensorflow-io-gcs-filesystem==0.37.1
termcolor==3.3.0
threadpoolctl==3.6.0
tomli==2.4.0
tomli_w==1.2.0
torch==2.4.1+cu124
torchaudio==2.4.1+cu124
torchvision==0.19.1+cu124
tqdm==4.67.3
triton==3.0.0
typer==0.20.1
typing_extensions==4.15.0
urllib3==2.6.3
Werkzeug==3.1.6
wrapt==2.1.2
```

## Thesis Chapters

[[Chapter 6 - aims|Ch. 6 — AIMS]]
