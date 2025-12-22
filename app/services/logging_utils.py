import logging
import os
from datetime import datetime

from config import DECISION_MAKING_LOGGING_FILE, ANSWER_GENERATION_LOGGING_FILE, LOG_MAX_LINES


class LineRotatingFileHandler(logging.Handler):
    """
    A simple line-based rotating file handler.

    - Writes to a base file path (e.g. logs/preprocess.log)
    - When the number of lines in the active file exceeds LOG_MAX_LINES,
      the current file is renamed to baseName{n}.ext (e.g. preprocess1.log,
      preprocess2.log, ...) and a fresh base file is created.
    - Existing numbered files are never deleted; the next index is always
      max(existing_indices) + 1. If no numbered files exist, rotation starts
      from 1.
    """

    def __init__(self, base_filepath: str, max_lines: int):
        super().__init__()
        self.base_filepath = base_filepath
        self.max_lines = max_lines

        base_dir = os.path.dirname(self.base_filepath)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        self.stream = open(self.base_filepath, "a", encoding="utf-8")
        self.current_lines = self._count_existing_lines()

    def _count_existing_lines(self) -> int:
        try:
            with open(self.base_filepath, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0

    def _get_next_index(self) -> int:
        """
        Look for files like baseName<N>.ext and return next integer index.
        If none exist, return 1.
        """
        directory, filename = os.path.split(self.base_filepath)
        name, ext = os.path.splitext(filename)

        max_index = 0
        if not directory:
            directory = "."

        try:
            for f in os.listdir(directory):
                if not f.startswith(name) or not f.endswith(ext):
                    continue
                # try to parse suffix between name and ext as int
                suffix = f[len(name) : len(f) - len(ext)]
                if not suffix:
                    # this is the base file (e.g. preprocess.log)
                    continue
                try:
                    idx = int(suffix)
                    if idx > max_index:
                        max_index = idx
                except ValueError:
                    continue
        except FileNotFoundError:
            # directory may not exist yet; treated as no files
            pass

        return max_index + 1 if max_index >= 0 else 1

    def _rotate(self):
        """Rotate the current base file to the next numbered file."""
        self.stream.close()

        directory, filename = os.path.split(self.base_filepath)
        name, ext = os.path.splitext(filename)
        next_index = self._get_next_index()
        rotated_name = f"{name}{next_index}{ext}"
        rotated_path = os.path.join(directory, rotated_name) if directory else rotated_name

        if os.path.exists(self.base_filepath):
            os.replace(self.base_filepath, rotated_path)

        self.stream = open(self.base_filepath, "a", encoding="utf-8")
        self.current_lines = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Count how many newlines we are about to write
            # We add one because logging frameworks typically append a newline.
            lines_to_add = msg.count("\n") + 1

            if self.current_lines + lines_to_add > self.max_lines:
                self._rotate()

            self.stream.write(msg + "\n")
            self.stream.flush()
            self.current_lines += lines_to_add
        except Exception:
            self.handleError(record)


def get_preprocess_logger() -> logging.Logger:
    logger = logging.getLogger("preprocess_logger")
    if logger.handlers:
        return logger

    handler = LineRotatingFileHandler(DECISION_MAKING_LOGGING_FILE, LOG_MAX_LINES)
    formatter = logging.Formatter(
        fmt="%(asctime)s | ip=%(ip)s | user_query=%(user_query)s | prompt=%(prompt)s | response=%(response)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_answer_generation_logger() -> logging.Logger:
    logger = logging.getLogger("answer_generation_logger")
    if logger.handlers:
        return logger

    handler = LineRotatingFileHandler(ANSWER_GENERATION_LOGGING_FILE, LOG_MAX_LINES)
    formatter = logging.Formatter(
        fmt="%(asctime)s | ip=%(ip)s | prompt=%(prompt)s | response=%(response)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
