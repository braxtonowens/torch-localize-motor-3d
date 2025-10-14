"""localizes flagellar-like motors in tomograms"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-localize-motor-3d")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Braxton Owens"
__email__ = "cbraxtonowens@gmail.com"

from .byu_motor_infer_direct import torch_localize_motor_3d

__all__ = ["torch_localize_motor_3d"]
