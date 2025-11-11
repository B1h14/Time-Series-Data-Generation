"""
Contains the setup for logging throughout the project.
"""
import logging
import sys
import os

def setup_logging(log_path : str = 'training.log') -> logging.Logger:
    """Configures the root logger."""
    
    # Define the log format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the lowest-level to capture everything

    # --- Create console handler (StreamHandler) ---
    # This logs to standard output (your terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Log INFO and above to the console
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # --- Create file handler (FileHandler) ---
    # This logs to a file
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)  # Log DEBUG and above to the file
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Add handlers to the root logger
    # Avoid adding handlers if they already exist
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger
