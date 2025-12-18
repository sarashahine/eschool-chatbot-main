from flask import Blueprint, request, jsonify, render_template
from qdrant_client.http import models as rest
import uuid

from flask import current_app
from config import BLOCK_THRESHOLD, BLOCK_MESSAGE, COLLECTION_NAME
from .utils import retrieve, truncate_answer_generation_history, generate_answer, pre_process_query, OllamaUnavailable

main_routes = Blueprint("main", __name__)

# -----------------------------
# HTML route
# -----------------------------
@main_routes.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Query route
# -----------------------------
@main_routes.route("/query", methods=["POST"])
def query():
    try:
        data = request.json
        user_query = data.get("query", "")
        history = data.get("history", [])
        unrelated_streak = int(data.get("unrelated_streak", 0))

        if not user_query:
            return jsonify({"error": "No query provided"}), 400



        decision = pre_process_query(
            user_query,
            current_app.decision_making_user_prompt,
            history,
            current_app.ollama_client,
            current_app.decision_making_system_prompt
        )

        requires_retrieval = (decision["category"] == "related")
        answer = decision["answer"]
        category = decision["category"]
        unrelated_streak = unrelated_streak + 1 if category == "unrelated" else 0

        if unrelated_streak >= BLOCK_THRESHOLD:
            return jsonify({"answer": BLOCK_MESSAGE, "blocked": True})

        if not requires_retrieval:
            history.append({"question": user_query, "answer": answer})
            return jsonify({"answer": answer, "blocked": False})

        results = retrieve(user_query, current_app.embedder, current_app.qdrant_client)
        context_block = "\n\n".join(
            f"Text: {r['text']}\nSection: {r['metadata'].get('section_title','')}\nURL: {r['metadata'].get('url','')}"
            for r in results
        )

        history = truncate_answer_generation_history(
            current_app.answer_generation_system_prompt_tokens,
            user_query,
            context_block,
            history,
            current_app.ollama_client
        )

        history_msgs = []
        for h in history:
            history_msgs.append({"role": "user", "content": h["question"]})
            history_msgs.append({"role": "assistant", "content": h["answer"]})

        user_prompt_for_answer_generation = f"Context:\n{context_block}\n\nQuestion:\n{user_query}\n\nAnswer:"
        answer = generate_answer(
            user_prompt_for_answer_generation,
            history_msgs,
            current_app.ollama_client,
            current_app.answer_generation_system_prompt
        )

        history.append({"question": user_query, "answer": answer})

        return jsonify({"answer": answer, "blocked": False, "unrelated_streak": unrelated_streak})

    except OllamaUnavailable:
        # Specific, user-friendly message when Ollama keeps failing
        return jsonify({
            "error": "The AI model is temporarily unavailable. Please try again in a little while."
        }), 503
    except Exception as e:
        # Fallback for any other unexpected server error
        return jsonify({"error": "Unexpected server error."}), 500

# -----------------------------
# Insert chunk
# -----------------------------
@main_routes.route("/chunks", methods=["POST"])
def insert_chunk():
    try:
        data = request.json or {}
        text = (data.get("text") or "").strip()
        metadata = data.get("metadata") or {}

        if not text:
            return jsonify({"error": "'text' is required."}), 400

        chunk_id = str(uuid.uuid4())
        vector = current_app.embedder.encode(text, convert_to_tensor=False).tolist()
        payload = {"text": text, **{k: v for k, v in metadata.items() if k in {"page_title", "url", "section_title"}}}

        point = rest.PointStruct(id=chunk_id, vector=vector, payload=payload)
        current_app.qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point], wait=True)

        return jsonify({"message": "Chunk inserted", "id": chunk_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Get a single chunk
# -----------------------------
@main_routes.route("/chunks/<int:chunk_id>", methods=["GET"])
def get_chunk(chunk_id):
    existing = current_app.qdrant_client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[chunk_id],
        with_vectors=False
    )
    print(existing)
    if not existing:
        return jsonify({"error": "Not found"}), 404

    payload = dict(existing[0].payload or {})
    return jsonify({
        "id": chunk_id,
        "text": payload.get("text", ""),
        "metadata": {
            "page_title": payload.get("page_title", ""),
            "url": payload.get("url", ""),
            "section_title": payload.get("section_title", ""),
        },
    })

# -----------------------------
# Update a chunk
# -----------------------------
@main_routes.route("/chunks/<int:chunk_id>", methods=["PUT"])
def update_chunk(chunk_id):
    try:
        data = request.json or {}
        new_text = data.get("text")
        metadata_updates = data.get("metadata")

        existing = current_app.qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[chunk_id],
            with_vectors=True
        )
        if not existing:
            return jsonify({"error": "Not found"}), 404

        payload = dict(existing[0].payload or {})
        vector = existing[0].vector

        if new_text:
            new_text = new_text.strip()
            payload["text"] = new_text
            vector = current_app.embedder.encode(new_text).tolist()

        if metadata_updates:
            payload.update({k: v for k, v in metadata_updates.items() if k in {"page_title", "url", "section_title"}})

        updated_point = rest.PointStruct(id=chunk_id, vector=vector, payload=payload)
        current_app.qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[updated_point], wait=True)

        return jsonify({"message": "Updated chunk"})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

# -----------------------------
# Delete a chunk
# -----------------------------
@main_routes.route("/chunks/<int:chunk_id>", methods=["DELETE"])
def delete_chunk(chunk_id):
    try:    
        current_app.qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=rest.PointIdsList(points=[chunk_id]),
            wait=True
        )
        return jsonify({"message": "Deleted chunk"})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
