# DeepCPI

## Usage

### 1) Install

```bash
conda create -n deep-cpi python=3.9
conda activate deep-cpi
# Install [PyTorch](https://pytorch.org/get-started/locally/)

git clone https://github.com/11Haoyu/nn-cpi
cd nn-cpi
pip install -e .
```

### 2) Training

```bash
python tools/train.py fit --config configs/gru_cpi_v1.yaml
python tools/train.py fit --config configs/gru_cpi_v2.yaml
python tools/train.py fit --config configs/gru_cpi_v3.yaml
python tools/train.py fit --config configs/gru_cpi_v4.yaml
python tools/train.py fit --config configs/lstm_cpi_v1.yaml
```

### 3) Tensorboard

```bash
tensorboard --logdir ./wandb
```
