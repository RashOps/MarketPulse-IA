import os
import logging
from logging.handlers import RotatingFileHandler

# Ensure logs directory exists
LOGS_DIR = "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = RotatingFileHandler(os.path.join(LOGS_DIR, "app.log"), maxBytes=1024*1024*5, backupCount=5)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger