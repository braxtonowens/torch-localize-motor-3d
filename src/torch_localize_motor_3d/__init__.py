"""localizes flagellar-like motors in tomograms"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-localize-motor-3d")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Braxton Owens"
__email__ = "cbraxtonowens@gmail.com"

from .download_weights import get_weights
from .torch_localize_motor_3d import predict_motor_location

__all__ = ["predict_motor_location", "get_weights"]
