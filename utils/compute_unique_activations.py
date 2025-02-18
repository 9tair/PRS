from utils.logger import global_logger  # Import from logger.py to avoid circular import
logger = global_logger  # Use the global logger

import numpy as np

def compute_unique_activations(activations):
    """
    Compute the number of unique activation patterns for the stored activations of the penultimate layer.

    Args:
        activations (numpy.ndarray): Activation maps retrieved from the penultimate layer.
    
    Returns:
        int: Number of unique activation patterns.
    """
    if activations is None or len(activations) == 0:
        logger.warning("No activations received in compute_unique_activations. Returning 0.")
        return 0

    logger.info(f"Computing unique activation patterns for {activations.shape[0]} samples...")

    decision_regions = set()   
    
    # Compute binary activation patterns (sign-based)
    signs = np.sign(activations)
    signs[signs == 0] = -1  

    # Debugging: Show the first few activation patterns
    logger.debug(f"First 5 activation rows (before binarization):\n{activations[:5]}")
    logger.debug(f"First 5 binarized patterns:\n{signs[:5]}")

    # Convert each row into a tuple for uniqueness tracking
    for row in signs:
        decision_regions.add(tuple(row))

    unique_count = len(decision_regions)
    logger.info(f"Total Unique Activation Patterns Found: {unique_count}")

    return unique_count