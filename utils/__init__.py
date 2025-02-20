from .dataset_loader import get_datasets
from .evaluate import evaluate
from .compute_unique_activations import compute_unique_activations
from .hooks import register_activation_hook
from .mr_tracker import compute_major_regions, save_major_regions
from .logger import setup_logger

# Explicitly define available imports
__all__ = [
    "get_datasets",
    "evaluate",
    "compute_unique_activations",
    "register_activation_hook",
    "compute_major_regions",
    "save_major_regions",
    "setup_logger",
]
