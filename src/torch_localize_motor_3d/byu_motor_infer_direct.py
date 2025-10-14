"""Direct inference wrapper for BYU motor localization using nnU-Net v2."""

from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, maximum_filter

# nnUNet v2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def _read_mrc(path: Union[str, Path]) -> np.ndarray:
    """Read MRC file and return as numpy array."""
    with mrcfile.open(path, permissive=True) as mrc:
        vol = np.asarray(mrc.data, dtype=np.float32)  # (Z,Y,X)
    if vol.ndim == 2:
        vol = vol[None, ...]
    return vol


def _resample_yx_to_longest_512(vol_zyx: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Match the common competition scaling: longest edge ~512, but only in-plane (Y,X).
    
    Returns resized volume and per-axis scales (sz, sy, sx) applied to go FROM original -> resized.
    """
    z, y, x = vol_zyx.shape
    longest = max(z, y, x)
    
    if longest <= 512 or longest == z:
        # keep Z fixed; only shrink Y/X if one of them is the longest
        scale = 512.0 / float(longest) if longest in (y, x) and longest > 512 else 1.0
        new_y = int(round(y * scale))
        new_x = int(round(x * scale))
    else:
        new_y, new_x = y, x
    
    if (new_y, new_x) == (y, x):
        return vol_zyx, (1.0, 1.0, 1.0)
    
    # torch interpolate expects (N,C,D,H,W); we use single-channel
    t = torch.from_numpy(vol_zyx[None, None])  # (1,1,Z,Y,X)
    t = F.interpolate(t, size=(z, new_y, new_x), mode="trilinear", align_corners=False)
    out = t[0, 0].numpy()
    
    return out, (1.0, new_y / y if y else 1.0, new_x / x if x else 1.0)


def _nms_3d_peaks(
    prob: np.ndarray, 
    threshold: float, 
    nms_radius: int = 3, 
    sigma: Optional[float] = 0.0
) -> np.ndarray:
    """Return (N,3) peaks in (z,y,x) index space from a 3D probability/heatmap."""
    p = prob
    if sigma and sigma > 0:
        p = gaussian_filter(p, sigma=sigma)
    
    mask = p >= threshold
    if not mask.any():
        return np.zeros((0, 3), dtype=np.float32)
    
    # 3D max-filter for local maxima
    mx = maximum_filter(p, size=(2 * nms_radius + 1,))
    peaks = (p == mx) & mask
    zyx = np.argwhere(peaks)
    
    # Sort by score (optional)
    scores = p[peaks]
    order = np.argsort(-scores)
    return zyx[order]


def torch_localize_motor_3d(
    mrc_path: Union[str, Path],
    ckpt_dir: Union[str, Path],
    *,
    folds: Union[str, Iterable[int]] = "all",
    checkpoint_name: str = "checkpoint_final.pth",
    threshold: float = 0.15,
    nms_radius: int = 3,
    smooth_sigma: float = 0.0,
    device: Optional[str] = None,
    resample_like_training: bool = True,
    return_int: bool = True,
) -> np.ndarray:
    """Run nnU-Net v2 model (from a trained folder) on an .mrc and return (N,3) (x,y,z) voxel coords
    in ORIGINAL MRC space.
    
    Parameters
    ----------
    mrc_path : Union[str, Path]
        Path to the MRC file
    ckpt_dir : Union[str, Path]
        Path to the nnU-Net training output folder (their weights dir)
    folds : Union[str, Iterable[int]], default "all"
        "all" or iterable of fold indices
    checkpoint_name : str, default "checkpoint_final.pth"
        Name of the checkpoint file
    threshold : float, default 0.15
        Threshold for peak detection
    nms_radius : int, default 3
        Radius for non-maximum suppression
    smooth_sigma : float, default 0.0
        Gaussian smoothing sigma (0 = no smoothing)
    device : Optional[str], default None
        Device to use (e.g., "cuda:0" or None for auto)
    resample_like_training : bool, default True
        Whether to resample to match training scale
    return_int : bool, default True
        Whether to return integer coordinates
        
    Returns
    -------
    np.ndarray
        (N,3) array of (x,y,z) coordinates in original MRC space
    """
    vol = _read_mrc(mrc_path)  # (Z,Y,X)
    orig_shape = np.array(vol.shape[::-1], np.float32)
    
    # Optional: mimic their training-scale policy so predictions line up
    if resample_like_training:
        vol_rs, (sz, sy, sx) = _resample_yx_to_longest_512(vol)
    else:
        vol_rs, (sz, sy, sx) = vol, (1.0, 1.0, 1.0)
    
    # nnU-Net wants channels-first 3D input; this solution was single-channel
    # shape for nnUNetPredictor: (C, Z, Y, X)
    x = vol_rs[None, ...]  # (1,Z,Y,X)
    
    # Initialize predictor from trained folder
    pred = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        # you can tweak parameters as needed; defaults are fine for most cases
    )
    pred.initialize_from_trained_model_folder(
        model_training_output_dir=str(ckpt_dir),
        use_folds=folds,
        checkpoint_name=checkpoint_name,
        device=device,  # e.g., "cuda:0" or None for auto
        verbose=False,
    )
    
    # Predict: returns logits/prob as (C, Z, Y, X). For regression-like heatmap, C==1.
    with torch.inference_mode():
        probs = pred.predict_from_ndarray(x)  # -> (C,Z,Y,X) numpy float32 in [0,1]
    
    heat = probs[0]  # single-channel
    
    # Find peaks in resized space (z,y,x)
    zyx_peaks = _nms_3d_peaks(
        heat, threshold=threshold, nms_radius=nms_radius, sigma=smooth_sigma
    )
    
    if zyx_peaks.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int32 if return_int else np.float32)
    
    # Map back to ORIGINAL space. We scaled Y/X by (sy, sx); Z left at sz (1.0 here).
    # Convert (z,y,x) -> (x,y,z) after scaling back.
    scale_back = np.array([1.0 / sx, 1.0 / sy, 1.0 / sz], np.float32)  # for (x,y,z)
    xyz_resized = np.stack(
        [zyx_peaks[:, 2], zyx_peaks[:, 1], zyx_peaks[:, 0]], axis=1
    ).astype(np.float32)
    xyz_orig = xyz_resized * scale_back[None, :]
    
    # Clip to bounds and cast
    xyz_orig[:, 0] = np.clip(xyz_orig[:, 0], 0, orig_shape[0] - 1)
    xyz_orig[:, 1] = np.clip(xyz_orig[:, 1], 0, orig_shape[1] - 1)
    xyz_orig[:, 2] = np.clip(xyz_orig[:, 2], 0, orig_shape[2] - 1)
    
    if return_int:
        xyz_orig = np.rint(xyz_orig).astype(np.int32)
    
    return xyz_orig