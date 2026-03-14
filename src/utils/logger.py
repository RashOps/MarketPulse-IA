import os
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

# Ensure logs directory exists
LOGS_DIR = "../../logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

def get_logger(name):
    logger = logging.getLogger(name)
    
    # Stop logger duplication if handlers already exist
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)
    
    # Path for log file
    log_file = Path("../..") / "logs" / "app.log"
    
    # If you want to create logs directory in current directory, use:
    # And change LOGS_DIR to "./logs"
    # log_file = Path("logs") / "app.log"
    # log_file.parent.mkdir(parents=True, exist_ok=True)

    # Optimize formatting for readability
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-6s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler (Only WARNING +)
    console_h = logging.StreamHandler()
    console_h.setLevel(logging.WARNING)
    console_h.setFormatter(formatter)

    # File Handler (Capture everything for audit)
    file_h = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(formatter)

    logger.addHandler(console_h)
    logger.addHandler(file_h)
    
    # Prevent log messages from being propagated to the root logger (avoid duplication)
    logger.propagate = False

    return logger