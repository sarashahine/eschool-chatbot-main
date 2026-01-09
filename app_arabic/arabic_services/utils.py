import json
import re
from uuid import UUID
from typing import Callable, TypeVar
import time

T = TypeVar("T")

class OllamaUnavailable(Exception):
    """Raised when Ollama keeps failing after retries."""
    pass

def load_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        print(f"Warning: {path} not found. Using empty prompt.")
        return ""

def call_ollama_with_retries(
    fn: Callable[[], T],
    max_seconds: float = 30.0,
    initial_delay: float = 1.0
) -> T:
    deadline = time.monotonic() + max_seconds
    delay = initial_delay
    last_exc = None

    while True:
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if time.monotonic() >= deadline:
                raise OllamaUnavailable(f"Ollama request failed after retries: {exc}") from exc
            time.sleep(delay)
            print("sleep: ", delay)
            delay = min(delay * 2, 5.0)

def is_valid_uuid(value: str) -> bool:
    try:
        UUID(value, version=4)
        return True
    except ValueError:
        return False

def safe_json(response: str):
    cleaned = re.sub(r"```json|```", "", response).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("No answer: ", response)
        return None
