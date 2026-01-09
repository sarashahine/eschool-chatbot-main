from flask import Blueprint, jsonify
from threading import Thread
from qdrant_client import QdrantClient

from ..nomic_services.nomic_ingestion import nomic_ingest_to_qdrant
from config import NOMIC_COLLECTION_NAME, QDRANT_HTTP

client = QdrantClient(url=QDRANT_HTTP)
nomic_admin_routes = Blueprint("nomic_admin_routes", __name__)


# -----------------------------
# Create Qdrant database (ingestion)
# -----------------------------

@nomic_admin_routes.route("/create-database", methods=["POST"])
def create_database():
    try:
        # Run ingestion in a background thread to avoid blocking Flask
        Thread(target=nomic_ingest_to_qdrant).start()
        return jsonify({
            "status": "started",
            "message": "Database ingestion started in background"
        }), 202
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

# -----------------------------
# Check database status
# -----------------------------
@nomic_admin_routes.route("/database-status", methods=["GET"])
def database_status():
    try:
        info = client.get_collection(NOMIC_COLLECTION_NAME)
        return jsonify({"exists": True, "info": info.dict()})
    except:
        return jsonify({"exists": False, "info": {}})

# -----------------------------
# Delete Qdrant database
# -----------------------------
@nomic_admin_routes.route("/delete-database", methods=["DELETE"])
def delete_database():
    try:
        client.delete_collection(NOMIC_COLLECTION_NAME)
        return jsonify({
            "status": "deleted",
            "message": f"Collection '{NOMIC_COLLECTION_NAME}' has been deleted."
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500