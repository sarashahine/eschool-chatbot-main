import logging
import os
from datetime import datetime

from config import LOG_DIR, LOG_FILE_BASENAME, LOG_FILE_EXT, LOG_MAX_LINES


class LineRotatingFileHandler(logging.Handler):
    def __init__(self, log_dir: str, base_name: str, ext: str, max_lines: int):
        super().__init__()
        self.log_dir = log_dir
        self.base_name = base_name
        self.ext = ext
        self.max_lines = max_lines
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        existing_file = self._find_existing_log(log_dir, base_name, ext)
        
        if existing_file:
            self.base_filepath = existing_file
            
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.base_filepath = os.path.join(
                log_dir, f"{base_name}_{timestamp}{ext}"
            )

        self.stream = open(self.base_filepath, "a", encoding="utf-8")

        self.current_lines = self._count_existing_lines()

    
    def _find_existing_log(self, directory: str, name: str, ext: str) -> str | None:

        files = os.listdir(directory)

        # Collect matching files
        matching_files = [
            f for f in files
            if f.startswith(f"{name}_") and f.endswith(ext)
        ]

        if not matching_files:
            return None
        
        # Sort newest first (timestamp is part of filename)
        matching_files.sort(reverse=True)

        newest_file = os.path.join(directory, matching_files[0])

        # Check if it still has space
        with open(newest_file, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)

        if line_count < self.max_lines:
            return newest_file
        return None


    def _count_existing_lines(self) -> int:
        try:
            with open(self.base_filepath, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
                return count
        except FileNotFoundError:
            return 0


    def _rotate(self):
        """Rotate the current base file to the next numbered file."""
        self.stream.close()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_filepath = os.path.join(
            self.log_dir, f"{self.base_name}_{timestamp}{self.ext}"
        )

        self.stream = open(self.base_filepath, "a", encoding="utf-8")
        self.current_lines = 0

    def emit(self, record: logging.LogRecord) -> None:
        
        msg = self.format(record)

        # Count how many newlines we are about to write
        # We add one because logging frameworks typically append a newline.
        lines_to_add = msg.count("\n") + 1

        if self.current_lines + lines_to_add > self.max_lines:
            self._rotate()
            
        self.stream.write(msg + "\n")
        self.stream.flush()
        self.current_lines += lines_to_add


# ---------------------------------------------------------------------
# Shared logger
# ---------------------------------------------------------------------

def get_chatbot_logger() -> logging.Logger:

    logger = logging.getLogger("website-chatbot-logger")

    if logger.handlers:
        return logger

    handler = LineRotatingFileHandler( LOG_DIR, LOG_FILE_BASENAME, LOG_FILE_EXT, LOG_MAX_LINES, )
    
    formatter = logging.Formatter(
        fmt=(
            "time=%(asctime)s\n"
            "ip=%(ip)s\n"
            "user_query=%(user_query)s\n"
            "stage=%(stage)s\n"
            "response=%(response)s\n"
            "embedding_model=%(embedding_model)s\n"
            "collection_name=%(collection_name)s\n"
            "retrieved_chunks=%(retrieved_chunks)s\n"
            "----------------------------------------"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


# ---------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------

def log_decision_making(
    *,
    ip: str,
    user_query: str,
    response: str,
    embedding_model,
    collection_name,
    retrieved_chunks,
):
    get_chatbot_logger().info(
        "",
        extra={
            "ip": ip,
            "user_query": user_query,
            "stage": "decision making",
            "response": response,
            "embedding_model": embedding_model,
            "collection_name": collection_name,
            "retrieved_chunks": retrieved_chunks,
        },
    )


def log_answer_generation(
    *,
    ip: str,
    user_query: str,
    response: str,
    embedding_model,
    collection_name,
    retrieved_chunks,
):
    get_chatbot_logger().info(
        "",
        extra={
            "ip": ip,
            "user_query": user_query,
            "stage": "answer generation",
            "response": response,
            "embedding_model": embedding_model,
            "collection_name": collection_name,
            "retrieved_chunks": retrieved_chunks,
        },
    )
