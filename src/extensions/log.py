"""
初始化日志
"""

import os
import time

from loguru import logger

from config import rootDir


def log_init():
    LOG_PATH = rootDir / "logs"
    os.makedirs(LOG_PATH, exist_ok=True)
    info_log_path = LOG_PATH / "info"
    error_log_path = LOG_PATH / "error"
    info_log_path.mkdir(exist_ok=True)
    error_log_path.mkdir(exist_ok=True)
    log_path_info = info_log_path / f'{time.strftime("%Y-%m-%d")}_info.log'
    log_path_error = error_log_path / f'{time.strftime("%Y-%m-%d")}_error.log'
    logger.add(str(log_path_error), rotation="12:00", retention="2 days", enqueue=True, level="ERROR")
    logger.add(str(log_path_info), rotation="12:00", retention="2 days", enqueue=True, level="DEBUG")
