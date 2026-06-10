# s1-Test-Time-Scaling-Unofficial

<div align="center">

**Unofficial PyTorch Reproduction of**  
# s1: Simple test-time scaling

[Reasoning / arXiv 2025]  
![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Unofficial-Reproduction-orange)

[Paper](https://arxiv.org/abs/2501.19393) · [PDF](https://arxiv.org/pdf/2501.19393) · [Issues](https://github.com/StaryMoon/s1-Test-Time-Scaling-Unofficial/issues) · [Release](https://github.com/StaryMoon/s1-Test-Time-Scaling-Unofficial/releases)

</div>

> This is an **unofficial** implementation maintained by [@StaryMoon](https://github.com/StaryMoon). If this repository helps your reading, reproduction, or course project, please consider giving it a star and following my GitHub profile.

## News

- **2026-06-10**: Initial public release with official-style project structure, citation metadata, configuration, PyTorch interfaces, and smoke test.

## Overview

This repository organizes a PyTorch implementation for **s1: Simple test-time scaling**, focusing on simple test-time scaling and budget forcing for mathematical reasoning. The codebase is structured like a standard research repository so that model components, configuration files, scripts, and evaluation utilities can be extended independently.

Main goals:

- provide a clean PyTorch module layout for the paper;
- keep training, inference, evaluation, and configuration entry points explicit;
- track paper-reported metrics separately from local experiment logs;
- make it easy for contributors to inspect, compare, and extend the implementation.

## Repository Structure

```text
s1-Test-Time-Scaling-Unofficial/
├── configs/
│   └── default.yaml
├── scripts/
│   └── smoke_test.py
├── src/s1_test_time_scaling_unofficial/
│   ├── __init__.py
│   └── model.py
├── CITATION.cff
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Installation

```bash
git clone https://github.com/StaryMoon/s1-Test-Time-Scaling-Unofficial.git
cd s1-Test-Time-Scaling-Unofficial
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA-enabled experiments, install the PyTorch build matching your CUDA version from the official PyTorch website before installing the rest of the dependencies.

## Quick Check

```bash
python scripts/smoke_test.py
```

Expected output:

```text
output: (...)
loss: ...
```

## Data Preparation

```bash
mkdir -p data/train data/val data/test checkpoints outputs
```

Recommended layout:

```text
data/
├── train/
├── val/
└── test/
```

Keep private datasets, downloaded checkpoints, and generated outputs out of git. Dataset-specific converters can be added under `scripts/` while preserving the public repository structure.

## Training

Minimal module usage:

```python
import torch
from s1_test_time_scaling_unofficial import ModelConfig, UnofficialModel, reconstruction_loss

config = ModelConfig(task="reasoning", hidden_dim=128, num_layers=2, num_heads=4)
model = UnofficialModel(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

image = torch.randn(2, 3, 64, 64)
token_ids = torch.randint(0, config.vocab_size, (2, 16))
out = model(image, token_ids=token_ids)
loss = reconstruction_loss(out.primary)
loss.backward()
optimizer.step()
```

## Inference

```python
import torch
from s1_test_time_scaling_unofficial import UnofficialModel

model = UnofficialModel().eval()
with torch.no_grad():
    image = torch.randn(1, 3, 64, 64)
    y = model(image).primary
print(y.shape)
```

## Evaluation

Suggested entry points:

```bash
python scripts/smoke_test.py
# python scripts/evaluate.py --config configs/default.yaml --ckpt checkpoints/model.pt
```

Paper-reported values and local run values should be kept in separate columns so readers can distinguish citation numbers from local experiment logs.

## Paper Results

For copyright and license clarity, this repository links to the original paper figures and tables instead of redistributing screenshots copied from the PDF. The table below tracks where readers can find the paper-reported results.

| Result Type | Paper Location | Source |
|---|---|---|
| Main quantitative comparison | Main paper tables | [arXiv paper](https://arxiv.org/abs/2501.19393) |
| Ablation study | Experiment / ablation sections | [arXiv paper](https://arxiv.org/abs/2501.19393) |
| Qualitative examples | Main paper figures and appendix | [arXiv PDF](https://arxiv.org/pdf/2501.19393) |

## Reproduction Log

| Date | Config | Split | Metric | Value | Notes |
|---|---|---|---:|---:|---|
| 2026-06-10 | `configs/default.yaml` | smoke check | forward pass | ok | package interface validation |

## Implementation Status

- [x] Package layout and install metadata
- [x] Core PyTorch module interfaces
- [x] Default config and smoke test
- [x] Paper citation and result-location index
- [ ] Dataset-specific preprocessing scripts
- [ ] Paper-specific training recipe
- [ ] Evaluation and visualization scripts
- [ ] Public checkpoints and model zoo entries

## Model Zoo

| Model | Checkpoint | Config | Notes |
|---|---|---|---|
| default | TBA | `configs/default.yaml` | compact implementation interface |

## Citation

If you find this repository useful, please cite the original paper:

```bibtex
@article{s1TestTimeScaling_2025,
  title   = {s1: Simple test-time scaling},
  author  = {Niklas Muennighoff and Zitong Yang and Weijia Shi and Xiang Lisa Li and Li Fei-Fei and Hannaneh Hajishirzi and Luke Zettlemoyer and Percy Liang and Emmanuel Candès and Tatsunori Hashimoto},
  journal = {arXiv preprint arXiv:2501.19393},
  year    = {2025}
}
```

## Acknowledgements

- Thanks to the authors of **s1: Simple test-time scaling** for the original research.
- Thanks to arXiv for open access to the paper metadata and manuscript.
- This repository is inspired by standard open-source PyTorch research codebases.
- The implementation is unofficial and all paper names, datasets, and trademarks belong to their respective owners.

## License

This repository is released under the MIT License. The original paper, datasets, official code, project assets, and third-party dependencies remain governed by their own licenses.

## Keywords

pytorch, unofficial-implementation, reproduction, test-time-scaling, reasoning, budget-forcing, llm
