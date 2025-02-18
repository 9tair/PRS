import logging
from utils.logger import global_logger  # Import from logger.py to avoid circular import

from train import train

if __name__ == "__main__":
    logger = global_logger  # Use the global logger

    try:
        logger.info("Starting training...")
        train()
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)