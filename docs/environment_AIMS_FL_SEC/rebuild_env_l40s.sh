python3 -m venv /workspace/venvs/aims
source /workspace/venvs/aims/bin/activate
python -m pip install --upgrade pip

pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

pip install tensorflow==2.18.0

pip install --no-deps \
  scikit-learn==1.7.0 \
  catboost==1.2.8 \
  pytorch-tabnet==4.1.0 \
  optuna==4.4.0 \
  plotly==6.1.2 \
  joblib==1.5.1 \
  tabulate==0.9.0 \
  "flwr[simulation]>=1.0.0" \
  "seaborn<0.13.0" \
  matplotlib \
  pandas

pip install pyparsing packaging pillow python-dateutil kiwisolver cycler contourpy fonttools
pip install scipy graphviz
pip install tqdm threadpoolctl
pip install alembic colorlog pyyaml sqlalchemy
pip install \
  "click>=8.0.0,<9.0.0" \
  "cryptography>=46.0.5,<47.0.0" \
  "grpcio-health-checking==1.70.0" \
  "iterators>=0.0.2,<0.0.3" \
  "pathspec>=0.12.1,<0.13.0" \
  "pycryptodome>=3.18.0,<4.0.0" \
  "tomli>=2.0.1,<3.0.0" \
  "tomli-w>=1.0.0,<2.0  \
  "rich>=13.5.0,<14.0.0"

pip uninstall -y grpcio-health-checking protobuf || true
pip install "protobuf==5.29.6" "grpcio-health-checking==1.70.0"

pip install "ray[default]>=2.10,<3"
