import json
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

# -----------------------------
# Configuration
# -----------------------------
JSON_PATH = r"C:\Users\ADMIN\Documents\GitHub\eschool-chatbot\json_files\all_embeddings.json"
COLLECTION_NAME = "docs"
QDRANT_HTTP = "http://localhost:6333"
EMBEDDING_MODEL = r"C:\Users\ADMIN\Documents\GitHub\eschool-chatbot\embeddinggemma\embeddinggemma-300m"

CHUNK_ID_TO_UPDATE = 23

# -----------------------------
# Load JSON and find chunk
# -----------------------------
def load_item_by_id(json_path, target_id):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for el in data:
        if isinstance(el, dict) and str(el.get("id")) == str(target_id):
            return {
                "id": int(el["id"]),
                "text": el.get("text", ""),
                "metadata": el.get("metadata", {})
            }
    return None

def main():
    print(f"Looking for chunk with ID {CHUNK_ID_TO_UPDATE}...")

    item = load_item_by_id(JSON_PATH, CHUNK_ID_TO_UPDATE)
    if not item:
        print(f"❌ Chunk with ID {CHUNK_ID_TO_UPDATE} not found.")
        return

    print("Loaded item:")
    print(item)

    # -----------------------------
    # Load model & embed
    # -----------------------------
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector = model.encode(item["text"], convert_to_tensor=False)
    vector = vector.tolist()  # ensure Python list

    # -----------------------------
    # Connect to Qdrant
    # -----------------------------
    client = QdrantClient(url=QDRANT_HTTP)

    # -----------------------------
    # Update point (upsert will overwrite existing)
    # -----------------------------
    updated_payload = {
        **item["metadata"],
        "text": item["text"],
        "url": "https://web.myeschoolhome.com/blog"  # example of new field
    }

    updated_point = rest.PointStruct(
        id=item["id"],
        vector=vector,           # optional: keep old vector or recompute
        payload=updated_payload
    )

    client.upsert(collection_name=COLLECTION_NAME, points=[updated_point], wait=True)
    print(f"✅ Updated chunk ID {item['id']} in Qdrant.")

    # -----------------------------
    # Verification
    # -----------------------------
    result = client.retrieve(collection_name=COLLECTION_NAME, ids=[item["id"]])
    print("\nRetrieved updated point from Qdrant:")
    print(result)

if __name__ == "__main__":
    main()
