import logging
import os
import sys

def setup_logger(name="app", log_file="logs/app_log.txt", level=logging.DEBUG):
    """Set up a global logger that works across multiple scripts."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if not logger.hasHandlers():
        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
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
        logger.propagate = True

    return logger

# Initialize a single global logger instance
global_logger = setup_logger("app", log_file="logs/app_log.txt")
