import json
import re
from uuid import UUID

from config import COLLECTION_NAME, NUM_RETRIEVED_CHUNKS, TOKEN_LIMIT, MODEL_NAME

# -----------------------------
# Prompt loade
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
        print("No answer")
        return None

def retrieve(query, embedder, qdrant_client):
    vector = embedder.encode(query).tolist()
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=NUM_RETRIEVED_CHUNKS,
    )
    points = getattr(results, "points", results)
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

def pre_process_query(query, history, ollama_client, preprocess_prompt):
    messages = [{"role": "system", "content": preprocess_prompt}]
    for h in history:
        messages.append({"role": "user", "content": h["question"]})
        messages.append({"role": "assistant", "content": h["answer"]})
    messages.append({"role": "user", "content": query})

    response = ollama_client.chat(model=MODEL_NAME, messages=messages, format="json")
    result = safe_json(response.message.content)

    if not result:
        return {"category": "related", "requires_retrieval": True, "direct_answer": None}

    category = result.get("category", "related")
    return {
        "category": category,
        "requires_retrieval": category == "related",
        "direct_answer": result.get("answer"),
    }

def truncate_history(history, user_query, context_block, ollama_client, system_prompt):
    def count_tokens(text):
        try:
            result = ollama_client.tokens(model=MODEL_NAME, prompt=text)
            return result.get("total_tokens", len(text.split()))
        except Exception:
            return len(text.split())

    total = count_tokens(system_prompt) + count_tokens(user_query) + count_tokens(context_block)
    kept = []

    for h in reversed(history):
        tokens = count_tokens(h["question"]) + count_tokens(h["answer"])
        if total + tokens > TOKEN_LIMIT:
            break
        kept.append(h)
        total += tokens

    return list(reversed(kept))

def generate_answer(user_prompt, history_msgs, ollama_client, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_msgs)
    messages.append({"role": "user", "content": user_prompt})

    response = ollama_client.chat(model=MODEL_NAME, messages=messages, format="json")
    try:
        content_json = json.loads(response.message.content)
        return content_json.get("answer", "")
    except Exception:
        return ""
