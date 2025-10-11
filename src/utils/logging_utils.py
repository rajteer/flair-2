import logging
import sys
from pathlib import Path

LOG_FORMATTER = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(threadName)s:%(name)s::  %(message)s",
)


def setup_logging(
    log_file: Path,
    log_formatter: logging.Formatter,
    *,
    no_stdout_logs: bool,
) -> None:
    """Set up logging to file and optionally to stdout."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if log_file.is_file():
        log_file.unlink()
    log_file_handler = logging.FileHandler(log_file, mode="a")
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    if not no_stdout_logs:
        log_stream_handler = logging.StreamHandler(sys.stdout)
        log_stream_handler.setLevel(logging.INFO)
        root_logger.addHandler(log_stream_handler)
