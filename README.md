# torch-localize-motor-3d

[![License](https://img.shields.io/pypi/l/torch-localize-motor-3d.svg?color=green)](https://github.com/braxtonowens/torch-localize-motor-3d/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-localize-motor-3d.svg?color=green)](https://pypi.org/project/torch-localize-motor-3d)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-localize-motor-3d.svg?color=green)](https://python.org)
[![CI](https://github.com/braxtonowens/torch-localize-motor-3d/actions/workflows/ci.yml/badge.svg)](https://github.com/braxtonowens/torch-localize-motor-3d/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/braxtonowens/torch-localize-motor-3d/branch/main/graph/badge.svg)](https://codecov.io/gh/braxtonowens/torch-localize-motor-3d)

A Python wrapper for localizing flagellar-like motors in tomograms using nnU-Net v2.

This package provides a simple interface to run inference on tomograms using the MIC_DKFZ motor localization solution from the BYU - Locating Bacterial Flagellar Motors 2025 Kaggle competition.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/braxtonowens/torch-localize-motor-3d.git
cd torch-localize-motor-3d
pip install -e torch_localize_motor_3d
```

```python
# 2. Run inference (weights download automatically on first use)
import mrcfile
from torch_localize_motor_3d import predict_motor_location

with mrcfile.open('your_tomogram.mrc', permissive=True) as mrc:
    coords = predict_motor_location(mrc.data)
    print(f'Found {len(coords)} motors:', coords)
```

Model weights are automatically downloaded from [Hugging Face](https://huggingface.co/braxtonowens/torch_localize_motor_3d_weights) on first use.

## Installation

### 1. Install the package

```bash
git clone https://github.com/braxtonowens/torch-localize-motor-3d.git
cd torch-localize-motor-3d
pip install -e torch_localize_motor_3d
```

### 2. Model weights (automatic)

Model weights are automatically downloaded from [Hugging Face](https://huggingface.co/braxtonowens/torch_localize_motor_3d_weights) on first use and cached in `~/.cache/torch_localize_motor_3d/checkpoints/`.

To pre-download weights:
```bash
python -m torch_localize_motor_3d.download_weights
```

Or in Python:
```python
from torch_localize_motor_3d import get_weights
get_weights()  # Downloads if not cached
```

### Requirements
- **CUDA-capable GPU (required)**
- Python ≥ 3.10
- PyTorch with CUDA support

## Usage

### Basic Usage

```python
import mrcfile
from torch_localize_motor_3d import predict_motor_location

# Read MRC file
with mrcfile.open('tomogram.mrc', permissive=True) as mrc:
    volume = mrc.data

# Predict motor locations
coords = predict_motor_location(volume)

print(f"Found {len(coords)} motor(s):")
for i, (x, y, z) in enumerate(coords):
    print(f"  Motor {i+1}: x={x}, y={y}, z={z}")
```

### Advanced Usage

```python
# Adjust detection parameters
coords = predict_motor_location(
    volume,
    threshold=0.15,      # Lower = more sensitive (more detections)
    nms_radius=3,        # Larger = fewer duplicate detections
    smooth_sigma=0.0,    # Try 1.0-2.0 if getting noisy detections
)
```

## Parameters

- `volume`: 3D numpy array (Z, Y, X) from MRC file
- `threshold`: Probability threshold for detection (default: 0.15, range: 0.0-1.0)
  - Lower values = more sensitive (more detections, more false positives)
  - Higher values = more specific (fewer detections, fewer false positives)
- `nms_radius`: Non-maximum suppression radius in voxels (default: 3)
  - Prevents duplicate detections of the same motor
  - Increase if motors are detected multiple times in close proximity
- `smooth_sigma`: Gaussian smoothing sigma (default: 0.0)
  - Set to 1.0-2.0 if getting noisy/duplicate detections
  - 0.0 = no smoothing
- `ckpt_dir`: Custom checkpoint directory (optional, uses default cache if None)

## Output

Returns a numpy array of shape `(N, 3)` where:
- N = number of detected motors
- Each row is `[x, y, z]` coordinates in the original volume space
- Coordinates are integers (voxel indices)

## Model Details

This package uses a trained nnU-Net v2 model from the MIC_DKFZ solution to the BYU - Locating Bacterial Flagellar Motors 2025 Kaggle competition. The model was trained on bacterial flagellar motor tomograms and achieves state-of-the-art detection performance.
