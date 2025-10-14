"""Direct inference wrapper for BYU motor localization using nnU-Net v2."""

from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, maximum_filter

# nnUNet v2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from .download_weights import get_weights, DEFAULT_CACHE_DIR as DEFAULT_CKPT_DIR


def _normalize_to_uint8_range(vol: np.ndarray) -> np.ndarray:
    """Normalize volume to [0, 255] range using percentile clipping.
    
    This matches the expected input range from JPEG stacks in the original implementation.
    """
    # Clip to 1st and 99th percentiles to handle outliers
    p1, p99 = np.percentile(vol, [1, 99])
    vol_clipped = np.clip(vol, p1, p99)
    
    # Rescale to [0, 255]
    vol_normalized = (vol_clipped - p1) / (p99 - p1) * 255.0
    
    return vol_normalized.astype(np.float32)


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
    filter_size = 2 * nms_radius + 1
    mx = maximum_filter(p, size=(filter_size, filter_size, filter_size))
    peaks = (p == mx) & mask
    zyx = np.argwhere(peaks)
    
    # Sort by score (optional)
    scores = p[peaks]
    order = np.argsort(-scores)
    return zyx[order]


def predict_motor_location(
    volume: np.ndarray,
    *,
    threshold: float = 0.15,
    nms_radius: int = 3,
    smooth_sigma: float = 0.0,
    ckpt_dir: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Predict flagellar motor locations in a 3D tomogram.
    
    Parameters
    ----------
    volume : np.ndarray
        3D numpy array (Z, Y, X) from MRC file. Use mrcfile.open() to read.
    threshold : float, default 0.15
        Probability threshold for peak detection. Lower values detect more motors.
    nms_radius : int, default 3
        Radius for non-maximum suppression. Prevents duplicate detections.
    smooth_sigma : float, default 0.0
        Gaussian smoothing sigma applied before peak detection (0 = no smoothing).
        Try 1.0-2.0 if getting noisy/duplicate detections.
    ckpt_dir : Optional[Union[str, Path]], default None
        Path to checkpoint directory. If None, automatically downloads from Hugging Face
        and caches in ~/.cache/torch_localize_motor_3d/checkpoints/
        
    Returns
    -------
    np.ndarray
        (N, 3) array of motor coordinates in (x, y, z) format, where N is the number
        of detected motors. Coordinates are in the original volume space.
        
    Examples
    --------
    >>> import mrcfile
    >>> with mrcfile.open('tomogram.mrc', permissive=True) as mrc:
    ...     volume = mrc.data
    >>> coords = predict_motor_location(volume)
    >>> print(f"Found {len(coords)} motors at: {coords}")
    
    Notes
    -----
    - Requires CUDA-capable GPU
    - First run will be slower due to model initialization
    - Model expects volumes in any value range (auto-normalized internally)
    """
    # Validate input
    if not isinstance(volume, np.ndarray):
        raise TypeError("volume must be a numpy array")
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D (Z,Y,X), got shape {volume.shape}")
    
    vol = volume.astype(np.float32)
    orig_shape = np.array(vol.shape[::-1], np.float32)  # (X, Y, Z)
    
    # Get checkpoint directory (auto-download if needed)
    if ckpt_dir is None:
        ckpt_dir = get_weights()
    else:
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    
    # Normalize volume to [0, 255] range to match training data
    vol = _normalize_to_uint8_range(vol)
    
    # Resize to match training scale
    vol_rs, (sz, sy, sx) = _resample_yx_to_longest_512(vol)
    
    # Use GPU (required)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but not available")
    device = torch.device("cuda:0")
    
    # Initialize predictor from trained folder (matching competition code)
    pred = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    pred.initialize_from_trained_model_folder(str(ckpt_dir), use_folds="all")
    
    # Convert to torch tensor and move to device
    img_tensor = torch.from_numpy(vol_rs).to(device).float()
    
    # Normalize like in the competition code
    img_tensor = (img_tensor - img_tensor.mean()) / img_tensor.std()
    
    # Predict using the actual competition API
    with torch.inference_mode():
        # predict_logits_from_preprocessed_data expects (C, Z, Y, X) input
        logits = pred.predict_logits_from_preprocessed_data(
            img_tensor[None],  # Add channel dimension
            out_device=device
        ).float()
        
        # Add batch dimension like in original code, then remove batch and channel
        logits = logits[None]  # Add batch dim
        probs = torch.sigmoid(logits)[0, 0]  # Remove batch and channel dims
    
    # Convert to numpy for peak detection
    heat = probs.cpu().numpy()
    
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
    
    # Clip to bounds and return as integers
    xyz_orig[:, 0] = np.clip(xyz_orig[:, 0], 0, orig_shape[0] - 1)
    xyz_orig[:, 1] = np.clip(xyz_orig[:, 1], 0, orig_shape[1] - 1)
    xyz_orig[:, 2] = np.clip(xyz_orig[:, 2], 0, orig_shape[2] - 1)
    
    return np.rint(xyz_orig).astype(np.int32)