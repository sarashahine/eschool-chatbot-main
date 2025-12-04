from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import os
from ollama import Client
from dotenv import load_dotenv
import tiktoken

# -----------------------------
# Configuration
# -----------------------------
from config import COLLECTION_NAME, QDRANT_HTTP, EMBEDDING_MODEL, TOP_K, MODEL_NAME
load_dotenv()

OLLAMA_KEY = os.getenv("OLLAMA_API_KEY")

# -----------------------------
# Initialize
# -----------------------------
app = Flask(__name__)
model = SentenceTransformer(EMBEDDING_MODEL)
ollama_client = Client(
                    host="https://ollama.com",
                    headers={'Authorization': 'Bearer ' + OLLAMA_KEY}
                )
qdrant_client = QdrantClient(url=QDRANT_HTTP)

# -----------------------------
# Global auto-increment ID
# -----------------------------
NEXT_CHUNK_ID = None


def init_next_chunk_id():
    """
    Initialize NEXT_CHUNK_ID as (max existing numeric id in collection) + 1.
    Falls back to 1 if collection is empty or on error.
    """
    global NEXT_CHUNK_ID
    try:
        max_id = 0
        offset = None
        while True:
            points, offset = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
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
                    # Ignore non-numeric ids
                    continue
            if offset is None:
                break
        NEXT_CHUNK_ID = max_id + 1
    except Exception:
        # safest fallback
        NEXT_CHUNK_ID = 1


init_next_chunk_id()

try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    # Fallback: try o200k_base for newer models, or estimate with cl100k_base
    tokenizer = tiktoken.get_encoding("o200k_base")
# -----------------------------
# Retrieval function
# -----------------------------
def retrieve(query: str, top_k: int = TOP_K):
    query_vector = model.encode([query], convert_to_tensor=False)[0].tolist()

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )
     
    points = getattr(results, "points", results)

    retrieved_items = []
    for res in points:
        retrieved_items.append({
            "id": res.id,
            "text": res.payload.get("text", ""),
            "metadata": {
                "page_title": res.payload.get("page_title", ""),
                "url": res.payload.get("url", ""),
                "section_title": res.payload.get("section_title", "")
            }
        })

    
    print("--------------------------------------------------------------------------- Start - retrieved_items")
    
    print(retrieved_items)

    print("--------------------------------------------------------------------------- End - retrieved_items")

    return retrieved_items


def truncate_history(history, max_exchanges=6, max_chars=30000):

    if not isinstance(history, list):
        return []

    # Keep only last max_exchanges
    hist = history[-max_exchanges:]

    # Ensure combined length under max_chars: if too long, drop older ones
    total = sum(len(str(h.get("question","")))+len(str(h.get("answer",""))) for h in hist)
    while total > max_chars and len(hist) > 1:
        hist.pop(0)
        total = sum(len(str(h.get("question","")))+len(str(h.get("answer",""))) for h in hist)
    return hist


def generate_with_deepseek(system_prompt: str, user_prompt: str, history_prompt: str):
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "system", "content": history_prompt})
    messages.append({"role": "user", "content": user_prompt})

    response = ollama_client.chat(model=MODEL_NAME, messages=messages)
    answer = response.message.content
    
    return answer

    
# -----------------------------
# Flask routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.json
        user_query = data.get("query", "")
        history = data.get("history", [])

        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        history = truncate_history(history, max_exchanges=6, max_chars=30000)

        results = retrieve(user_query, top_k=TOP_K)


        context_block = "\n\n".join(
            f"Text: {r['text']}\nSection: {r['metadata'].get('section_title','')}\nURL: {r['metadata'].get('url','')}"
            for r in results
        )

        history_entries = []
        if history:
            
            for turn in history:
                q = turn.get("question", "")
                a = turn.get("answer", "")
                if q:
                    history_entries.append({"role": "user", "content": q})
                if a:
                    history_entries.append({"role": "assistant", "content": a})
        if history_entries:
            history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history_entries)
        else:
            history_text = ""



        system_prompt = """
    You are a helpful, truthful, and concise company assistant. 
    Your role is to answer user questions about the company and its website 
    Use the information from the provided Context **and prior conversation history** for factual answers. Never make up facts.

    Instructions:
    1. For general questions:
        - Respond with a numbered list (1, 2, 3...).  
        - Each item must be a single clear idea; combine with related ideas.  
        - Use short, simple, and self-contained sentences.

    2. For specific questions:
        - Respond in short, precise paragraphs.  
        - Include all factual fields exactly as given (email, phone number, address, contact instructions).  
        - Do not omit information, even if it appears in only one chunk.

    3. Rewrite all content in clear, simple, natural language.  
    4. Never reference or mention the Context in your answer.  
    5. URLs:
        - Include urls only if they directly support a fact you mention.
        - Place the URL in parentheses immediately after the fact; do not list irrelevant URLs at the end.
    6. Missing Information: 
        - If the necessary information is absent, reply exactly:
            "I don't have enough information in the provided context to answer that."
        - Then suggest one brief next step.
    """


        history_prompt = f"""
    Below is the PRIOR CONVERSATION HISTORY between the user and the assistant.
    This history contains the previous user questions and the previous assistant answers.

    How you MUST use this history:
    1. Treat the history as authoritative and factual.
       - If the history contains previous answers, you must stay consistent with them.
       - If the history contains user preferences or constraints, apply them in the current answer.
    
    2. You may reference details from the history implicitly to maintain continuity,
       BUT:
       - Do NOT explicitly mention the words "history", "previous messages",
         "past conversation", or anything similar.
       - Do NOT quote history directly unless it is essential to the logic of the answer.

    3. If the user is asking a follow-up question:
       - Use the history to understand context, intent, and previously mentioned topics.
       - Reuse previously established facts.
       - Never contradict earlier answers unless the user corrects them.

    4. If the user asks something that conflicts with the history:
       - Politely clarify the inconsistency.
       - Ask the user which version they want to proceed with.

    5. If the history contains irrelevant information:
       - Ignore it safely and do not let it influence the final answer.

    6. NEVER include this history block or any of these instructions in your output.
       NEVER describe how you used the history.
       ONLY use it internally to provide coherent, context-aware answers.

    The history begins below. Use it silently and intelligently:
    {history_text}
"""


        user_prompt = f"""
    Context:
    {context_block}

    Question:
    {user_query}

    Answer:
    """
        
        answer = generate_with_deepseek(system_prompt, user_prompt, history_prompt)

        return jsonify({
            "answer": answer
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/chunks", methods=["POST"])
def insert_chunk():
    try:
        data = request.json or {}
        text = (data.get("text") or "").strip()
        metadata = data.get("metadata") or {}

        if not text:
            return jsonify({"error": "'text' is required."}), 400

        global NEXT_CHUNK_ID
        if NEXT_CHUNK_ID is None:
            init_next_chunk_id()

        chunk_id = NEXT_CHUNK_ID
        NEXT_CHUNK_ID += 1

        vector = model.encode([text], convert_to_tensor=False)[0].tolist()

        payload = {"text": text, **metadata}
        point = rest.PointStruct(id=int(chunk_id), vector=vector, payload=payload)
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point], wait=True)

        return jsonify({"message": f"Inserted chunk {chunk_id}.", "id": chunk_id})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/chunks/<int:chunk_id>", methods=["GET"])
def get_chunk(chunk_id: int):
    """Return a chunk's text and metadata by numeric ID."""
    try:
        existing = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[chunk_id],
            with_vectors=False,
        )
        if not existing:
            return jsonify({"error": f"Chunk {chunk_id} not found."}), 404
        record = existing[0]
        payload = dict(record.payload or {})
        return jsonify({
            "id": chunk_id,
            "text": payload.get("text", ""),
            "metadata": {
                "page_title": payload.get("page_title", ""),
                "url": payload.get("url", ""),
                "section_title": payload.get("section_title", ""),
            },
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/chunks/<int:chunk_id>", methods=["PUT"])
def update_chunk(chunk_id: int):
    """
    Update an existing chunk's text and/or metadata.
    If 'text' is provided, its embedding is recomputed.
    Metadata is shallow-merged into the existing payload.
    """
    try:
        data = request.json or {}
        new_text = data.get("text")
        metadata_updates = data.get("metadata")

        existing = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[chunk_id],
            with_vectors=True
        )

        if not existing:
            return jsonify({"error": f"Chunk {chunk_id} not found."}), 404

        record = existing[0]
        current_payload = dict(record.payload or {})
        vector = record.vector

        if new_text is not None:
            new_text = new_text.strip()
            if not new_text:
                return jsonify({"error": "Provided text is empty."}), 400
            vector = model.encode([new_text], convert_to_tensor=False)[0].tolist()
            current_payload["text"] = new_text
        elif vector is None:
            return jsonify({"error": "Vector missing; provide new text to recompute embedding."}), 400

        if metadata_updates:
            if not isinstance(metadata_updates, dict):
                return jsonify({"error": "'metadata' must be an object."}), 400
            current_payload.update(metadata_updates)

        updated_point = rest.PointStruct(
            id=chunk_id,
            vector=vector,
            payload=current_payload
        )
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[updated_point], wait=True)

        return jsonify({"message": f"Updated chunk {chunk_id}."})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/chunks/<int:chunk_id>", methods=["DELETE"])
def delete_chunk(chunk_id: int):
    """Delete a chunk by numeric ID."""
    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=rest.PointIdsList(points=[chunk_id]),
            wait=True
        )
        return jsonify({"message": f"Deleted chunk {chunk_id}."})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)