<h1 align="center">Neural Low-Discrepancy Sequences</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2510.03745">Paper</a> • ICML 2026
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

## Overview
This repository contains the code for our ICML 2026 paper *Neural Low-Discrepancy Sequences*. It provides PyTorch implementations for model training, discrepancy-based fine‑tuning, and reproducible experiments.

<p align="center">
  <img src="img/diagram.png" alt="NeuroLDS architecture" width="600"/>
</p>

*Figure: Overview of the NeuroLDS architecture.*


## Key files & scripts
- `scripts/models.py` — Main model definition (NeuroLDS).
- `scripts/main.py` — Create/generate sequences; trains/evaluates NeuroLDS.
- `scripts/utils.py` — Aid/utility functions (discrepancy losses, seeding, I/O, plotting).
- `scripts/hypertuning.py` — Hyperparameter optimization via Optuna.
- `scripts/smoke_test.sh` — Sanity check; run to verify the setup works end-to-end.

## Installation
```bash
# Recommended: Python 3.10–3.12 (tested on 3.11)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick check
Run the smoke test to verify that everything is set up correctly:
```bash
cd scripts
bash smoke_test.sh
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
