"""Download model weights from Hugging Face."""

import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, list_repo_files


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "torch_localize_motor_3d" / "checkpoints"
HF_REPO = "braxtonowens/torch_localize_motor_3d_weights"

# Required checkpoint filenames (will search for them in the repo)
REQUIRED_FILES = [
    "checkpoint_final.pth",
    "dataset.json",
    "dataset_fingerprint.json",
    "plans.json",
]


def get_weights(cache_dir: Optional[Path] = None, force: bool = False) -> Path:
    """Download model weights from Hugging Face if not already cached.
    
    Parameters
    ----------
    cache_dir : Optional[Path]
        Directory to cache weights. If None, uses default cache location.
    force : bool
        If True, re-download even if weights exist.
        
    Returns
    -------
    Path
        Path to the checkpoint directory.
        
    Raises
    ------
    RuntimeError
        If download fails.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    cache_dir = Path(cache_dir)
    
    # nnU-Net expects files in fold_all/ subdirectory
    fold_dir = cache_dir / "fold_all"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if all files exist
    all_exist = all((fold_dir / f).exists() for f in REQUIRED_FILES)
    
    if all_exist and not force:
        print(f"Using cached weights from: {cache_dir}")
        return cache_dir
    
    print(f"Downloading model weights from Hugging Face...")
    print(f"Repository: {HF_REPO}")
    print(f"Destination: {cache_dir}")
    print()
    
    # List all files in the repo to find where our files are
    print("Scanning HuggingFace repo for checkpoint files...")
    try:
        all_files = list_repo_files(repo_id=HF_REPO)
    except Exception as e:
        raise RuntimeError(f"Failed to list repo files: {e}")
    
    # Find the path for each required file
    file_paths = {}
    for required in REQUIRED_FILES:
        matching = [f for f in all_files if f.endswith(required)]
        if not matching:
            raise RuntimeError(f"Could not find {required} in HuggingFace repo")
        if len(matching) > 1:
            # Prefer files in fold_all if multiple matches
            fold_all_matches = [f for f in matching if "fold_all" in f]
            file_paths[required] = fold_all_matches[0] if fold_all_matches else matching[0]
        else:
            file_paths[required] = matching[0]
    
    print(f"Found checkpoint files:")
    for name, path in file_paths.items():
        print(f"  {name}: {path}")
    print()
    
    # Download each file
    for filename, hf_path in file_paths.items():
        dest_path = fold_dir / filename
        
        if dest_path.exists() and not force:
            print(f"Skipping (already exists): {filename}")
            continue
        
        print(f"Downloading: {filename}")
        try:
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=hf_path,
            )
            # Copy to our local structure (fold_all subdirectory)
            shutil.copy2(downloaded_path, dest_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download {hf_path}: {e}")
    
    print()
    print("✓ Download complete!")
    print(f"Weights saved to: {cache_dir}")
    
    return cache_dir


def main():
    """CLI entry point for downloading weights."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download torch_localize_motor_3d model weights"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if weights exist",
    )
    
    args = parser.parse_args()
    
    try:
        get_weights(cache_dir=args.cache_dir, force=args.force)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
