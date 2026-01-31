import logging
import os
from datetime import datetime


def setup_file_logger(log_dir="logs", name="experiment"):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        log_dir, f"{name}_{timestamp}.log"
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear() 
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger, log_path
