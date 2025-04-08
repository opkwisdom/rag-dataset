import torch
from argparse import ArgumentParser
import os
import logging
from datetime import datetime

def set_config():
    config = {}
    config['model_path'] = "psyche/KoT5"
    config['en_temp'] = '문단:'
    config['de_temp'] = '이 문단의 핵심 문구:'
    config["position_factor"] = 1.2e8
    config["length_factor"] = 1.1
    config['max_len'] = 512
    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['save_mode'] = False
    return config

def setup_logger(log_dir, log_filename="app.log", log_level=logging.INFO):
    """Setup a logger with console and file handlers.

    Args:
        log_dir (str): Directory to store log files.
        log_filename (str): Log file name.
        log_level (int): Logging level, e.g., logging.DEBUG, logging.INFO, etc.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename_with_time = f"{log_filename}_{timestamp}.log"

    logger = logging.getLogger(log_filename_with_time)
    logger.setLevel(log_level)

    # Formatter for the logs
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename_with_time))
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger