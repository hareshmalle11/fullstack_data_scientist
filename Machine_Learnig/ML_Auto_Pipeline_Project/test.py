from src.logger import logging
from src.exception import CustomException
import sys

logging.info("Logger is working correctly")

try:
    x = 1 / 0
except Exception as e:
    raise CustomException(e, sys)