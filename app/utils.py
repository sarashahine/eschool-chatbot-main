import json
import re
from uuid import UUID
import time
from typing import Callable, TypeVar

from config import COLLECTION_NAME, NUM_RETRIEVED_CHUNKS, TOKEN_LIMIT, MODEL_NAME

# -----------------------------
# Prompt loader
# -----------------------------
def load_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        print(f"Warning: {path} not found. Using empty prompt.")
        return ""

# -----------------------------
# Utilities
# -----------------------------

T = TypeVar("T")

class OllamaUnavailable(Exception):
    """Raised when Ollama keeps failing after retries."""
    pass

def call_ollama_with_retries(
    fn: Callable[[], T],
    max_seconds: float = 30.0,      # total time to keep retrying
    initial_delay: float = 1.0      # first sleep between retries
) -> T:
    deadline = time.monotonic() + max_seconds
    delay = initial_delay
    last_exc = None

    while True:
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            # stop retrying once we've exceeded our time budget
            if time.monotonic() >= deadline:
                raise OllamaUnavailable(
                    f"Ollama request failed after retries: {exc}"
                ) from exc
            # wait and try again (simple exponential backoff)
            time.sleep(delay)
            print("sleep: ", delay)
            delay = min(delay * 2, 5.0)


def is_valid_uuid(value):
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

def retrieve(query, embedder, qdrant_client):
    vector = embedder.encode(query).tolist()
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=NUM_RETRIEVED_CHUNKS,
    )
    points = getattr(results, "points")
    return [
        {
            "text": p.payload.get("text", ""),
            "metadata": {
                "page_title": p.payload.get("page_title", ""),
                "url": p.payload.get("url", ""),
                "section_title": p.payload.get("section_title", ""),
            },
        }
        for p in points
    ]

def pre_process_query(query, decision_making_user_prompt, history, ollama_client, decision_making_system_prompt):
    messages = [{"role": "system", "content": decision_making_system_prompt}]

    for h in history:
        messages.append({"role": "user", "content": h["question"]})
        messages.append({"role": "assistant", "content": h["answer"]})
    
    user_prompt_with_query = decision_making_user_prompt.replace("{{USER_QUERY}}", query)
    messages.append({"role": "user", "content": user_prompt_with_query})

    response = call_ollama_with_retries(
        lambda: ollama_client.chat(model=MODEL_NAME, messages=messages, format="json")
    )

    result = safe_json(response.message.content)

    if not result:
        return {"category": "related", "answer": query}

    category = result.get("category", "related")
    answer = result.get("answer", query)
    
    return {
        "category": category,
        "answer": answer,
    }

def count_tokens(text, ollama_client):
        try:
            result = ollama_client.tokens(model=MODEL_NAME, prompt=text)
            return result.get("total_tokens", len(text.split()))
        except Exception:
            return len(text.split())

def truncate_history(history, user_query, context_block, ollama_client, answer_generation_system_prompt_tokens):

    total = answer_generation_system_prompt_tokens + count_tokens(user_query,ollama_client) + count_tokens(context_block,ollama_client)
    kept = []

    for h in reversed(history):
        tokens = count_tokens(h["question"],ollama_client) + count_tokens(h["answer"],ollama_client)
        if total + tokens > TOKEN_LIMIT:
            break
        kept.append(h)
        total += tokens

    return list(reversed(kept))

def generate_answer(user_prompt, history_msgs, ollama_client, answer_generation_system_prompt):
    messages = [{"role": "system", "content": answer_generation_system_prompt}]
    messages.extend(history_msgs)
    messages.append({"role": "user", "content": user_prompt})

    response = call_ollama_with_retries(
        lambda: ollama_client.chat(model=MODEL_NAME, messages=messages, format="json")
    )
    
    try:
        content_json = json.loads(response.message.content)
        return content_json.get("answer", "")
    except Exception: return ""