from flask import Flask, request, jsonify, render_template
import json
import os
import re
from dotenv import load_dotenv
from ollama import Client
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models as rest

from config import (
    COLLECTION_NAME,
    QDRANT_HTTP,
    EMBEDDING_MODEL,
    TOP_K,
    MODEL_NAME,
    TOKEN_LIMIT,
)

# -----------------------------
# Initialization
# -----------------------------
load_dotenv()

app = Flask(__name__)
model = SentenceTransformer(EMBEDDING_MODEL)

OLLAMA_KEY = os.getenv("OLLAMA_API_KEY")
if not OLLAMA_KEY:
    raise RuntimeError("Missing OLLAMA_API_KEY")

ollama_client = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
)

qdrant_client = QdrantClient(url=QDRANT_HTTP)

# -----------------------------
# Chunk ID management
# -----------------------------
NEXT_CHUNK_ID = None


def init_next_chunk_id():
    global NEXT_CHUNK_ID
    max_id = 0
    offset = None

    while True:
        points, offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=False,
            with_vectors=False,
            offset=offset,
        )
        if not points:
            break

        for p in points:
            try:
                max_id = max(max_id, int(p.id))
            except Exception:
                pass

        if offset is None:
            break

    NEXT_CHUNK_ID = max_id + 1


init_next_chunk_id()

# -----------------------------
# Prompt loading
# -----------------------------
def load_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


SYSTEM_PROMPT_BASE = load_prompt("prompts/system_prompt.txt")
PREPROCESS_PROMPT_BASE = load_prompt("prompts/preprocess_prompt.txt")

# -----------------------------
# Helpers
# -----------------------------
def retrieve(query: str, top_k: int):
    vector = model.encode([query], convert_to_tensor=False)[0].tolist()

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
    )

    points = getattr(results, "points", results)

    return [
        {
            "text": p.payload.get("text", ""),
            "section": p.payload.get("section_title", ""),
            "url": p.payload.get("url", ""),
        }
        for p in points
    ]


def safe_json_load(raw: str):
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def pre_process_query(user_query, history):
    messages = [{"role": "system", "content": PREPROCESS_PROMPT_BASE}]

    for turn in history or []:
        if turn.get("question"):
            messages.append({"role": "user", "content": turn["question"]})
        if turn.get("answer"):
            messages.append({"role": "assistant", "content": turn["answer"]})

    messages.append({"role": "user", "content": user_query})

    response = ollama_client.chat(
        model=MODEL_NAME,
        messages=messages,
        format={
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": ["related", "unrelated"]},
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["category", "answer", "confidence"],
        },
    )

    result = safe_json_load(response.message.content) or {}

    category = result.get("category", "related")
    requires_retrieval = category == "related"

    return {
        "requires_retrieval": requires_retrieval,
        "direct_answer": result.get("answer") if not requires_retrieval else None,
        "retrieval_query": user_query if requires_retrieval else None,
        "category": category,
    }


def count_tokens(text: str) -> int:
    try:
        res = ollama_client.tokens(model=MODEL_NAME, prompt=text)
        return res.get("total_tokens", len(text.split()))
    except Exception:
        return len(text.split())


def truncate_history(history, user_query, context_block):
    total = (
        count_tokens(SYSTEM_PROMPT_BASE)
        + count_tokens(user_query)
        + count_tokens(context_block)
    )

    kept = []
    for h in reversed(history or []):
        tokens = count_tokens(h.get("question", "")) + count_tokens(h.get("answer", ""))
        if total + tokens > TOKEN_LIMIT:
            break
        kept.append(h)
        total += tokens

    return list(reversed(kept))


def generate_answer(system_prompt, user_prompt, history):
    messages = [{"role": "system", "content": system_prompt}]

    for h in history:
        messages.append(h)

    messages.append({"role": "user", "content": user_prompt})

    response = ollama_client.chat(model=MODEL_NAME, messages=messages)

    try:
        parsed = json.loads(response.message.content)
        return parsed.get("answer", "")
    except Exception:
        return ""

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    data = request.json or {}
    user_query = data.get("query", "").strip()
    history = data.get("history", [])
    unrelated_streak = int(data.get("unrelated_streak", 0))

    if not user_query:
        return jsonify({"error": "Query required"}), 400

    decision = pre_process_query(user_query, history)
    category = decision["category"]

    unrelated_streak = unrelated_streak + 1 if category == "unrelated" else 0

    if unrelated_streak >= 2000:
        return jsonify(
            {
                "answer": "For more info visit our website.",
                "blocked": True,
                "unrelated_streak": unrelated_streak,
            }
        )

    if not decision["requires_retrieval"]:
        answer = decision["direct_answer"]
    else:
        docs = retrieve(decision["retrieval_query"], TOP_K)

        context = "\n\n".join(
            f"Text: {d['text']}\nSection: {d['section']}\nURL: {d['url']}"
            for d in docs
        )

        history = truncate_history(history, user_query, context)

        history_msgs = []
        for h in history:
            history_msgs.append({"role": "user", "content": h["question"]})
            history_msgs.append({"role": "assistant", "content": h["answer"]})

        user_prompt = f"Context:\n{context}\n\nQuestion:\n{user_query}\n\nAnswer:"
        answer = generate_answer(SYSTEM_PROMPT_BASE, user_prompt, history_msgs)

    history.append({"question": user_query, "answer": answer})

    return jsonify(
        {
            "answer": answer,
            "history": history,
            "blocked": False,
            "unrelated_streak": unrelated_streak,
        }
    )


# -----------------------------
# Server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
