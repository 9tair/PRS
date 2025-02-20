import logging
import os
import sys
from datetime import datetime

def setup_logger(model_name, dataset_name, batch_size, log_dir="logs/", level=logging.DEBUG):
    """
    Set up a dynamic logger that creates a unique log file for each experiment setting.
    
    Args:
        model_name (str): Name of the model being used (e.g., CNN-6, VGG16).
        dataset_name (str): Dataset name (e.g., CIFAR10, MNIST).
        batch_size (int): Batch size for training.
        log_dir (str): Directory where logs will be stored.
        level (int): Logging level (default: DEBUG).
    
    Returns:
        logger: Configured logger instance.
    """

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename based on experiment settings & timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{model_name}_{dataset_name}_batch_{batch_size}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Create logger
    logger = logging.getLogger(f"{model_name}_{dataset_name}_batch_{batch_size}")
    logger.setLevel(level)

    # Prevent duplicate handlers
    if not logger.hasHandlers():
        # Create file handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)

        # Create console handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)

        # Define formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        # Enable propagation to ensure logs show in the main script
        logger.propagate = False  # Prevent duplicate logs if used across modules

    return logger