# torch-localize-motor-3d

[![License](https://img.shields.io/pypi/l/torch-localize-motor-3d.svg?color=green)](https://github.com/braxtonowens/torch-localize-motor-3d/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-localize-motor-3d.svg?color=green)](https://pypi.org/project/torch-localize-motor-3d)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-localize-motor-3d.svg?color=green)](https://python.org)
[![CI](https://github.com/braxtonowens/torch-localize-motor-3d/actions/workflows/ci.yml/badge.svg)](https://github.com/braxtonowens/torch-localize-motor-3d/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/braxtonowens/torch-localize-motor-3d/branch/main/graph/badge.svg)](https://codecov.io/gh/braxtonowens/torch-localize-motor-3d)

A Python wrapper for localizing flagellar-like motors in tomograms using nnU-Net v2.

This package provides a simple interface to run inference on MRC files using the BYU motor localization solution from the Kaggle competition.

## Installation

```bash
# Clone and install in development mode
git clone https://github.com/braxtonowens/torch-localize-motor-3d.git
cd torch-localize-motor-3d
pip install -e .
```

## Usage

### Basic Usage

```python
from torch_localize_motor_3d import torch_localize_motor_3d

# Run inference on an MRC file
coords = torch_localize_motor_3d(
    "my_volume.mrc",
    ckpt_dir="/path/to/your_trained_model_folder",
    folds="all",                 # or (0,1,2,3,4)
    threshold=0.15,
    nms_radius=3,
    smooth_sigma=0.0,            # try 1.0–2.0 if peaks are noisy
    device="cuda:0",             # or "cpu" if no GPU
)

print(f"Found {len(coords)} motor coordinates:")
print(coords[:10])  # First 10 coordinates (x, y, z)
```

### Example Script

Run the provided example script:

```bash
python example_usage.py
```

Make sure to update the `ckpt_dir` path in the script to point to your trained nnU-Net model directory.

### Test MRC Reading

To test if MRC file reading works correctly:

```bash
python test_mrc_reading.py
```

## Parameters

- `mrc_path`: Path to the MRC file
- `ckpt_dir`: Path to the nnU-Net training output folder
- `folds`: "all" or iterable of fold indices (default: "all")
- `threshold`: Threshold for peak detection (default: 0.15)
- `nms_radius`: Radius for non-maximum suppression (default: 3)
- `smooth_sigma`: Gaussian smoothing sigma, 0 = no smoothing (default: 0.0)
- `device`: Device to use, e.g., "cuda:0" or "cpu" (default: None for auto)
- `resample_like_training`: Whether to resample to match training scale (default: True)
- `return_int`: Whether to return integer coordinates (default: True)

## Requirements

- Python ≥ 3.10
- PyTorch
- NumPy
- SciPy
- mrcfile
- nnU-Net v2 (installed from the BYU solution repository)

## Model Requirements

You need a trained nnU-Net v2 model from the BYU motor localization solution. The model should be trained on the bacterial flagellar motor dataset and saved in the standard nnU-Net output format.

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork braxtonowens/torch-localize-motor-3d --clone
# or just
# gh repo clone braxtonowens/torch-localize-motor-3d
cd torch-localize-motor-3d
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
