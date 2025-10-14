#!/usr/bin/env python3
"""Example usage script for torch_localize_motor_3d."""

from pathlib import Path

from torch_localize_motor_3d import torch_localize_motor_3d


def main():
    """Run example inference on local.mrc file."""
    # Path to the MRC file
    mrc_path = Path("local.mrc")
    
    # You'll need to update this path to point to your trained model
    # This should be the output directory from nnU-Net training
    ckpt_dir = "/path/to/your_trained_model_folder"
    
    if not mrc_path.exists():
        print(f"Error: {mrc_path} not found!")
        return
    
    if not Path(ckpt_dir).exists():
        print(f"Error: Model checkpoint directory {ckpt_dir} not found!")
        print("Please update the ckpt_dir path to point to your trained nnU-Net model.")
        return
    
    print(f"Running inference on {mrc_path}...")
    
    try:
        coords = torch_localize_motor_3d(
            mrc_path,
            ckpt_dir=ckpt_dir,
            folds="all",                 # or (0,1,2,3,4)
            threshold=0.15,
            nms_radius=3,
            smooth_sigma=0.0,            # try 1.0–2.0 if peaks are noisy
            device="cuda:0",             # or "cpu" if no GPU
        )
        
        print(f"Found {len(coords)} motor coordinates:")
        print("First 10 coordinates (x, y, z):")
        print(coords[:10])
        
        # Save results to file
        output_file = "motor_coordinates.txt"
        with open(output_file, "w") as f:
            f.write("# Motor coordinates (x, y, z)\n")
            for coord in coords:
                f.write(f"{coord[0]}\t{coord[1]}\t{coord[2]}\n")
        
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Make sure you have:")
        print("1. A valid trained nnU-Net model in the checkpoint directory")
        print("2. PyTorch with CUDA support (if using GPU)")
        print("3. All required dependencies installed")


if __name__ == "__main__":
    main()