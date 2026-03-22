#!/usr/bin/env bash
set -e

python3 -m venv /workspace/venvs/aims
source /workspace/venvs/aims/bin/activate
python -m pip install --upgrade pip

# Ajuste aqui com base no que efetivamente funcionou nessa máquina:
# Exemplo:
# pip install scikit-learn==1.7.0 catboost==1.2.8 pytorch-tabnet==4.1.0 optuna==4.4.0 plotly==6.1.2 joblib==1.5.1 tabulate==0.9.0 matplotlib pandas seaborn

# Se houver torch/tensorflow nessa máquina, registre aqui também.
