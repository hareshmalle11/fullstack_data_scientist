import logging
import sys
from voice_assistant.config.settings import Settings

def setup_logger():
    logger = logging.getLogger("voice_assistant")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if logger is configured multiple times
    if logger.handlers:
        return logger

    # Log format
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    try:
        file_handler = logging.FileHandler(Settings.LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file handler at {Settings.LOG_FILE}: {e}")

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger

# Globally available logger instance
logger = setup_logger()
