import logging
import sys
from utils.logger import setup_logger  
from project_root.cnn6 import train

if __name__ == "__main__":
    logger = setup_logger("MAIN", "ALL", "NA")  # Main logger for general errors

    try:
        logger.info("Starting training...")
        logger.info(f"Python Version: {sys.version}")
        train()
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)