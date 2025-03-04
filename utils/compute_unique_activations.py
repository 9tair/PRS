from utils.logger import setup_logger  # Import dynamic logger
import numpy as np

def compute_unique_activations(activations, logger):
    """
    Compute the number of unique activation patterns for the stored activations of the penultimate layer.

    Args:
        activations (numpy.ndarray): Activation maps retrieved from the penultimate layer.
        logger: Logger instance for tracking logs.

    Returns:
        int: Number of unique activation patterns.
    """
    logger.info(f"Received activations shape: {activations.shape}")

    if activations is None or len(activations) == 0:
        logger.warning("No activations received in compute_unique_activations. Returning 0.")
        return 0

    num_samples, num_activations = activations.shape
    logger.info(f"Computing unique activation patterns for {num_samples} samples...")
    logger.info(f"Each sample has {num_activations} activations.")

    decision_regions = set()  # Store unique activation patterns   
    
    # Compute binary activation patterns (sign-based)
    signs = np.sign(activations)
    signs[signs == 0] = -1  # Convert zeroes to -1 to maintain consistency

    # Debugging: Show the first few activation patterns
    logger.debug(f"First activation row (before binarization):\n{activations[0]}")
    logger.debug(f"First binarized pattern:\n{signs[0]}")

    # Convert each row into a tuple for uniqueness tracking
    for row in signs:
        decision_regions.add(tuple(row))

    unique_count = len(decision_regions)
    logger.info(f"Total Unique Activation Patterns Found: {unique_count}")

    return unique_count
