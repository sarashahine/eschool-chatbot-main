from flask import Blueprint, request, jsonify
from qdrant_client.http import models as rest
import requests

from flask import current_app
from config import NOMIC_COLLECTION_NAME, NOMIC_EMBED_URL, NOMIC_MODEL_NAME

nomic_chunk_routes = Blueprint("nomic_chunk_routes", __name__)

# -----------------------------
# Insert chunk
# -----------------------------
@nomic_chunk_routes.route("/chunks", methods=["POST"])
def insert_chunk():
    try:
        data = request.json or {}
        text = (data.get("text") or "").strip()
        metadata = data.get("metadata") or {}

        if not text:
            return jsonify({"error": "'text' is required."}), 400

        response = requests.post(
            NOMIC_EMBED_URL,
            json={
                "model": NOMIC_MODEL_NAME,
                "prompt": text,
            },
            timeout=30,
        )
        response.raise_for_status()

        vector = response.json()["embedding"]

        payload = {
            "text": text,
            **{k: v for k, v in metadata.items() if k in {"page_title", "url", "section_title"}},
        }

        # Let Qdrant auto-assign the point ID by omitting the `id` field.
        point = rest.PointStruct(vector=vector, payload=payload)
        current_app.qdrant_client.upsert(
            collection_name=NOMIC_COLLECTION_NAME,
            points=[point],
            wait=True,
        )

        return jsonify({"message": "Chunk inserted"})

    except requests.RequestException as e:
        return jsonify({"error": "Embedding service error", "details": str(e)}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Get a single chunk
# -----------------------------
@nomic_chunk_routes.route("/chunks/<int:chunk_id>", methods=["GET"])
def get_chunk(chunk_id):
    existing = current_app.qdrant_client.retrieve(
        collection_name=NOMIC_COLLECTION_NAME,
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
@nomic_chunk_routes.route("/chunks/<int:chunk_id>", methods=["PUT"])
def update_chunk(chunk_id):
    try:
        data = request.json or {}
        new_text = data.get("text")
        metadata_updates = data.get("metadata")

        existing = current_app.qdrant_client.retrieve(
            collection_name=NOMIC_COLLECTION_NAME,
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

            response = requests.post(
                NOMIC_EMBED_URL,
                json={
                    "model": NOMIC_MODEL_NAME,
                    "prompt": new_text,
                },
                timeout=30,
            )
            response.raise_for_status()

            vector = response.json()["embedding"]

        if metadata_updates:
            payload.update({k: v for k, v in metadata_updates.items() if k in {"page_title", "url", "section_title"}})

        updated_point = rest.PointStruct(id=chunk_id, vector=vector, payload=payload)
        current_app.qdrant_client.upsert(collection_name=NOMIC_COLLECTION_NAME, points=[updated_point], wait=True)

        return jsonify({"message": "Updated chunk"})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

# -----------------------------
# Delete a chunk
# -----------------------------
@nomic_chunk_routes.route("/chunks/<int:chunk_id>", methods=["DELETE"])
def delete_chunk(chunk_id):
    try:    
        current_app.qdrant_client.delete(
            collection_name=NOMIC_COLLECTION_NAME,
            points_selector=rest.PointIdsList(points=[chunk_id]),
            wait=True
        )
        return jsonify({"message": "Deleted chunk"})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
