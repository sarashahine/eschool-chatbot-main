import json
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import TEXT_TO_EMBED_PATH, COLLECTION_NAME, BATCH_SIZE, QDRANT_HTTP, EMBEDDING_MODEL, DISTANCE,VECTOR_SIZE, NORMALIZE_EMBEDDINGS


# -----------------------------
# Helper functions
# -----------------------------
def load_items(json_path: str) -> List[Dict]:

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for idx, el in enumerate(data):
        if isinstance(el, dict):
            text = el.get("text", "")
            meta = el.get("metadata", {})
            items.append({"id": idx, "text": text, "metadata": meta})
        else:
            items.append({"id": idx, "text": str(el), "metadata": {}})
    return items


def upsert_batch(client: QdrantClient, items: List[Dict], model: SentenceTransformer):

    texts = [item["text"] for item in items]
    ids = [item["id"] for item in items]

    vectors = model.encode(texts, batch_size=len(texts), convert_to_numpy=True, normalize_embeddings=NORMALIZE_EMBEDDINGS)


    points = [
        rest.PointStruct(
            id=ids[i],
            vector=vectors[i],
            payload={"text": texts[i], **items[i]["metadata"],},
        )
        for i in range(len(ids))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)


# -----------------------------
# Main execution
# -----------------------------
def main():

    items = load_items(TEXT_TO_EMBED_PATH)

    model = SentenceTransformer(EMBEDDING_MODEL)

    client = QdrantClient(url=QDRANT_HTTP)

    try:
        client.get_collection(COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance[DISTANCE.upper()]),
        )

    for i in tqdm(range(0, len(items), int(BATCH_SIZE)), desc="Upserting"):
        batch = items[i : i + BATCH_SIZE]
        upsert_batch(client, batch, model)

    print(f"Ingestion completed. Total items: {len(items)}")


if __name__ == "__main__":
    main()
