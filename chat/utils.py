from functools import partial
import logging
import os
import time
import typing as t

import loguru
import loguru._logger
from memoization import cached, CachingAlgorithmFlag
from settings import Settings


def _filter_logs(record: dict) -> bool:
    if record["level"].no <= 10 and not Settings.basic_settings.log_verbose:
        return False
    if record["level"].no == 40 and not Settings.basic_settings.log_verbose:
        record["exception"] = None
    return True


@cached(max_size=100, algorithm=CachingAlgorithmFlag.LRU)
def build_logger(log_file: str = "chatchat"):
    loguru.logger._core.handlers[0]._filter = _filter_logs
    logger = loguru.logger.opt(colors=True)
    logger.opt = partial(loguru.logger.opt, colors=True)
    logger.warn = logger.warning

    if log_file:
        if not log_file.endswith(".log"):
            log_file = f"{log_file}.log"
        if not os.path.isabs(log_file):
            log_file = str((Settings.basic_settings.LOG_PATH / log_file).resolve())
        logger.add(log_file, colorize=False, filter=_filter_logs)

    return logger


logger = logging.getLogger(__name__)


class LoggerNameFilter(logging.Filter):
    def filter(self, record):
        return True


def get_log_file(log_path: str, sub_dir: str):
    log_dir = os.path.join(log_path, sub_dir)
    os.makedirs(log_dir, exist_ok=False)
    return os.path.join(log_dir, f"{sub_dir}.log")


def get_config_dict(
        log_level: str, log_file_path: str, log_backup_count: int, log_max_bytes: int
) -> dict:
    log_file_path = (
        log_file_path.encode("unicode-escape").decode()
        if os.name == "nt"
        else log_file_path
    )
    log_level = log_level.upper()
    config_dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "formatter": {
                "format": (
                    "%(asctime)s %(name)-12s %(process)d %(levelname)-8s %(message)s"
                )
            },
        },
        "filters": {
            "logger_name_filter": {
                "()": __name__ + ".LoggerNameFilter",
            },
        },
        "handlers": {
            "stream_handler": {
                "class": "logging.StreamHandler",
                "formatter": "formatter",
                "level": log_level,
            },
            "file_handler": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "formatter",
                "level": log_level,
                "filename": log_file_path,
                "mode": "a",
                "maxBytes": log_max_bytes,
                "backupCount": log_backup_count,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "chatchat_core": {
                "handlers": ["stream_handler", "file_handler"],
                "level": log_level,
                "propagate": False,
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["stream_handler", "file_handler"],
        },
    }
    return config_dict


def get_timestamp_ms():
    t = time.time()
    return int(round(t * 1000))
