# qGAN
qGAN for loading random distributions based on Zoufal 2019 Paper. 2025 Fall PKU Intro to QC Course Project

# Environment Setup

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```
# Train qGAN

```bash
python lognorm.py
```

This will train qGAN in local simulator, and output final weights.

# Sample Trained VQC on IBM Quantum Platform

First, paste the output weight into `run.py`, and fill in the IBM Quantum Platform API Key.

```
python run.py
```
