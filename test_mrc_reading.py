#!/usr/bin/env python3
"""Test script to verify MRC file reading works correctly."""

from pathlib import Path

import numpy as np

from torch_localize_motor_3d.byu_motor_infer_direct import _read_mrc


def main():
    """Test MRC file reading."""
    mrc_path = Path("local.mrc")
    
    if not mrc_path.exists():
        print(f"Error: {mrc_path} not found!")
        return
    
    print(f"Testing MRC file reading on {mrc_path}...")
    
    try:
        vol = _read_mrc(mrc_path)
        print(f"Successfully read MRC file!")
        print(f"Shape: {vol.shape}")
        print(f"Data type: {vol.dtype}")
        print(f"Value range: [{vol.min():.3f}, {vol.max():.3f}]")
        print(f"Mean: {vol.mean():.3f}")
        print(f"Std: {vol.std():.3f}")
        
        # Check if it's 3D
        if vol.ndim == 3:
            print(f"Volume dimensions (Z, Y, X): {vol.shape}")
        else:
            print(f"Warning: Expected 3D volume, got {vol.ndim}D")
            
    except Exception as e:
        print(f"Error reading MRC file: {e}")


if __name__ == "__main__":
    main()