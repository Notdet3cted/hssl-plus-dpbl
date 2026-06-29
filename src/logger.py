import logging
import os
import yaml
from datetime import datetime

def setup_logger(name="WESAD_HSSL_DPBL", log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Determine log directory: arg > env var > config file > fallback
        if log_dir is None:
            log_dir = os.environ.get("HSSL_DPBL_LOG_DIR")
        if log_dir is None:
            try:
                with open("config/config.yaml", "r") as f:
                    cfg = yaml.safe_load(f)
                log_dir = cfg.get("paths", {}).get("logs", "logs/")
            except Exception:
                log_dir = "logs/"

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"experiment_{timestamp}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        logger.info(f"Log file: {log_path}")

    return logger

if __name__ == "__main__":
    log = setup_logger()
    log.info("Logger initialized.")

