# File: src/utils/logging_utils.py
import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    experiment_name: str,
    log_level: int = logging.INFO,
    log_dir: str = "outputs/logs",
    log_filename_prefix: str = "training_log"
) -> logging.Logger:
    """Set up logging for the experiment.

    Logs will be saved to a file and also printed to stdout.

    Args:
        experiment_name (str): Name of the current experiment, used in log filename.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        log_dir (str): Directory to save log files.
        log_filename_prefix (str): Prefix for the log file name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger('VectorFloorSeg')
    logger.setLevel(log_level)

    # Prevent multiple handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Define log file name
    log_file = log_path / f"{log_filename_prefix}_{experiment_name}.log"

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

if __name__ == '__main__':
    # Example usage:
    test_logger = setup_logging(experiment_name="test_experiment", log_level=logging.DEBUG)
    test_logger.debug("This is a debug message.")
    test_logger.info("This is an info message.")
    test_logger.warning("This is a warning message.")
    test_logger.error("This is an error message.")
    print(f"Test log file created in {Path('outputs/logs') / 'training_log_test_experiment.log'}")
