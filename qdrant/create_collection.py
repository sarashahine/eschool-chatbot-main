"""
  - Load JSON [{"id":..., "text":..., "metadata":...}]
  - Embed using embeddinggemma
  - Upsert into Qdrant in batches
"""

import json
import os
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv

# -----------------------------
# Configuration - define here
# -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

JSON_PATH = r"C:\Users\Lenovo\Downloads\project\eschool-chatbot-main\text_to_embed.json"               # path to your JSON file
COLLECTION_NAME = "docs_api"               # Qdrant collection name
BATCH_SIZE = 1                            # batch size for upsert
QDRANT_HTTP = "http://localhost:6333"       # HTTP control URL
VECTOR_SIZE = 768
DISTANCE = "COSINE"                         # vector distance metric


# -----------------------------
# Helper functions
# -----------------------------
def load_items(json_path: str) -> List[Dict]:
    """Load JSON and return list of dicts with 'id', 'text', 'metadata'."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for idx, el in enumerate(data):
        if isinstance(el, dict):
            text = el.get("text", "")
            source_id = el.get("id", str(idx))
            meta = el.get("metadata", {})  # keep page_title, url, section_title intact
            items.append({"id": int(source_id), "text": text, "metadata": meta})
        else:
            items.append({"id": str(idx), "text": str(el), "metadata": {}})
    return items

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Call Gemini embedding API to get embeddings for a list of texts."""
    client = genai.Client(api_key=GOOGLE_API_KEY)
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=VECTOR_SIZE)
    )
    embeddings = [emb.values for emb in result.embeddings]
    return embeddings

def upsert_batch(client: QdrantClient, collection_name: str, items: List[Dict]):
    """Embed texts and upsert into Qdrant in one batch."""
    texts = [item["text"] for item in items]
    ids = [item["id"] for item in items]
    metas = [item.get("metadata", {}) for item in items]

    vectors = embed_texts(texts)

    points = [
        rest.PointStruct(
            id=ids[i],
            vector=vectors[i],
            payload={"text": texts[i], **metas[i]},  # merge text and metadata
        )
        for i in range(len(ids))
        # vector=vectors[i], ??? only this
    ]
    client.upsert(collection_name=collection_name, points=points, wait=True)
    # or client.upsert(collection_name=collection_name, points=[point]) ???


# -----------------------------
# Main execution
# -----------------------------
def main():
    # Load items
    items = load_items(JSON_PATH)
    print(f"Loaded {len(items)} items from {JSON_PATH}")

    # Connect to Qdrant
    client = QdrantClient(url=QDRANT_HTTP)

    # Ensure collection exists
    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance[DISTANCE.upper()]),
        )

    # Upsert in batches
    for i in tqdm(range(0, len(items), BATCH_SIZE), desc="Upserting"):
        batch = items[i : i + BATCH_SIZE]
        upsert_batch(client, COLLECTION_NAME, batch)

    print(f"Ingestion completed. Total items: {len(items)}")


if __name__ == "__main__":
    main()
