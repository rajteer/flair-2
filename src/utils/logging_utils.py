import logging
import sys
from pathlib import Path


class ResilientFileHandler(logging.FileHandler):
    """A FileHandler that handles NFS stale file handle errors.

    This handler flushes after every log record and automatically recovers
    from OSError 116 (Stale file handle) by reopening the file.
    This is crucial for HPC environments with NFS filesystems.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record, handling stale file handle errors."""
        try:
            super().emit(record)
            self.flush()
        except OSError as e:
            if e.errno == 116:  # Stale file handle
                self._reopen_stream()
                try:
                    super().emit(record)
                    self.flush()
                except OSError:
                    pass  # Give up silently to avoid crashing the program
            else:
                raise

    def _reopen_stream(self) -> None:
        """Close and reopen the stream to recover from stale handle."""
        try:
            if self.stream:
                self.stream.close()
        except OSError:
            pass
        self.stream = self._open()


# Keep alias for backwards compatibility
FlushingFileHandler = ResilientFileHandler


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
