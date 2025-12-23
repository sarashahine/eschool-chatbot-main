import logging
import os
from datetime import datetime

from config import LOGGING_FILE, LOG_MAX_LINES


class LineRotatingFileHandler(logging.Handler):
    def __init__(self, base_filepath: str, max_lines: int):
        super().__init__()
        self.max_lines = max_lines
        directory, filename = os.path.split(base_filepath)
        name, ext = os.path.splitext(filename)

        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        if not os.path.exists(base_filepath):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.base_filepath = os.path.join(directory, f"{name}_{timestamp}{ext}")
        else:
            self.base_filepath = base_filepath

        self.stream = open(self.base_filepath, "a", encoding="utf-8")
        self.current_lines = self._count_existing_lines()


    def _count_existing_lines(self) -> int:
        try:
            with open(self.base_filepath, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0

    def _get_next_index(self) -> int:
        directory, filename = os.path.split(self.base_filepath)
        print("base_filepath: ", self.base_filepath)

        print("directory: ", directory)
        print("filename: ", filename)
        
        name, ext = os.path.splitext(filename)

        print("name: ", name)
        print("ext: ", ext)

        max_index = 0
        directory = directory or "."

        for f in os.listdir(directory):
            if f.startswith(name) and f.endswith(ext):
                suffix = f[len(name):-len(ext)]
                if suffix.startswith("_") and suffix[1:].isdigit():
                    max_index = max(max_index, int(suffix[1:]))

        return max_index + 1


    def _rotate(self):
        """Rotate the current base file to the next numbered file."""
        self.stream.close()

        directory, filename = os.path.split(self.base_filepath)
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_path = os.path.join(directory, f"{name}_{timestamp}{ext}")

        os.replace(self.base_filepath, rotated_path)

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
    logger = logging.getLogger("chatbot_logger")
    if logger.handlers:
        return logger

    handler = LineRotatingFileHandler(
        LOGGING_FILE,
        LOG_MAX_LINES,
    )

    formatter = logging.Formatter(
        fmt=(
            "time=%(asctime)s\n"
            "ip=%(ip)s\n"
            "user_query=%(user_query)s\n"
            "stage=%(stage)s\n"
            "response=%(response)s\n"
            "prompt=%(prompt)s\n"
            "history=%(history)s\n"
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
    prompt,
    history: str = "",
):
    get_chatbot_logger().info(
        "",
        extra={
            "ip": ip,
            "user_query": user_query,
            "stage": "decision making",
            "response": response,
            "prompt": prompt,
            "history": history,
        },
    )


def log_answer_generation(
    *,
    ip: str,
    user_query: str,
    response: str,
    prompt,
    history: str = "",
):
    get_chatbot_logger().info(
        "",
        extra={
            "ip": ip,
            "user_query": user_query,
            "stage": "answer generation",
            "response": response,
            "prompt": prompt,
            "history": history,
        },
    )
