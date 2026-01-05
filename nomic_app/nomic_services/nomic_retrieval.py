import json
import requests

from config import NOMIC_COLLECTION_NAME, NUM_RETRIEVED_CHUNKS, TOKEN_LIMIT, MODEL_NAME, NOMIC_EMBED_URL, NOMIC_MODEL_NAME
from nomic_app.nomic_services.utils import call_ollama_with_retries, safe_json
from nomic_app.nomic_services.nomic_logging_utils import log_decision_making

def retrieve(query, qdrant_client):
    # 1. Get embedding from local Ollama (Nomic)
    response = requests.post(
        NOMIC_EMBED_URL,
        json={
            "model": NOMIC_MODEL_NAME,
            "prompt": query
        }
    )
    response.raise_for_status()
    vector = response.json()["embedding"]
    
    # 2. Query Qdrant
    results = qdrant_client.query_points(
        collection_name=NOMIC_COLLECTION_NAME,
        query=vector,
        limit=NUM_RETRIEVED_CHUNKS,
    )
    points = getattr(results, "points")

    # 3. Format results
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

def pre_process_query(query, decision_making_user_prompt, history, ollama_client, decision_making_system_prompt, user_ip):
    messages = [{"role": "system", "content": decision_making_system_prompt}]

    for h in history:
        messages.append({"role": "user", "content": h["question"]})
        messages.append({"role": "assistant", "content": h["answer"]})

    user_prompt_with_query = decision_making_user_prompt.replace("{{USER_QUERY}}", query)
    messages.append({"role": "user", "content": user_prompt_with_query})

    response = call_ollama_with_retries(
        lambda: ollama_client.chat(model=MODEL_NAME, messages=messages, format="json")
    )

    try:
        raw_response = getattr(getattr(response, "message", None), "content", str(response))
        log_decision_making(
                ip = user_ip,
                user_query = query,
                response = raw_response,
        )
    except Exception:
        # Logging should never break the main flow
        pass

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

def truncate_answer_generation_history(fixed_prompt_tokens, user_query, context_block, history, ollama_client):
    total = fixed_prompt_tokens + count_tokens(user_query, ollama_client) + count_tokens(context_block, ollama_client)
    kept = []

    for h in reversed(history):
        tokens = count_tokens(h["question"], ollama_client) + count_tokens(h["answer"], ollama_client)
        if total + tokens > TOKEN_LIMIT:
            break
        kept.append(h)
        total += tokens

    return list(reversed(kept))
