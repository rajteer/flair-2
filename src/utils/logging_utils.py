import logging
import sys
from pathlib import Path


class FlushingFileHandler(logging.FileHandler):
    """A FileHandler that flushes after every log record.

    This is crucial for HPC environments where processes may be killed
    unexpectedly, ensuring no log messages are lost in the buffer.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record and immediately flush the stream."""
        super().emit(record)
        self.flush()


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
    # Use FlushingFileHandler to ensure logs are written immediately
    # This prevents log loss when HPC jobs are killed unexpectedly
    log_file_handler = FlushingFileHandler(log_file, mode="a")
    log_file_handler.setFormatter(log_formatter)
    root_logger.addHandler(log_file_handler)

    if not no_stdout_logs:
        log_stream_handler = logging.StreamHandler(sys.stdout)
        log_stream_handler.setLevel(logging.INFO)
        log_stream_handler.setFormatter(log_formatter)
        root_logger.addHandler(log_stream_handler)
