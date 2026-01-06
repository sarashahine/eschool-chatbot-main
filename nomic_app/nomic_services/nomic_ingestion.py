import json
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm
import requests

from config import TEXT_TO_EMBED_PATH, NOMIC_COLLECTION_NAME, BATCH_SIZE, QDRANT_HTTP, DISTANCE, VECTOR_SIZE, NORMALIZE_EMBEDDINGS, NOMIC_EMBED_URL, NOMIC_MODEL_NAME

def load_items(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for idx, el in enumerate(data):
        if isinstance(el, dict):
            items.append({
                "id": idx,
                "text": el.get("text", ""),
                "metadata": el.get("metadata", {}),
            })
        else:
            items.append({"id": idx, "text": str(el), "metadata": {}})
    return items

def upsert_batch(client: QdrantClient, items: List[Dict]):
    texts = [item["text"] for item in items]
    ids = [item["id"] for item in items]

    embeddings = []

    for text in texts:
        response = requests.post(
            NOMIC_EMBED_URL,
            json={
                "model": NOMIC_MODEL_NAME,
                "prompt": text,  # Ollama supports batch prompts
            },
            timeout=60,
        )
        response.raise_for_status()
        embeddings.append(response.json()["embedding"])

    points = [
        rest.PointStruct(
            id=ids[i],
            vector=embeddings[i],
            payload={"text": texts[i], **items[i]["metadata"]},
        )
        for i in range(len(ids))
    ]

    client.upsert(
        collection_name=NOMIC_COLLECTION_NAME,
        points=points,
        wait=True,
    )

def nomic_ingest_to_qdrant():
    items = load_items(TEXT_TO_EMBED_PATH)
    client = QdrantClient(url=QDRANT_HTTP)

    try:
        client.get_collection(NOMIC_COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=NOMIC_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance[DISTANCE.upper()],
            ),
        )

    for i in tqdm(range(0, len(items), int(BATCH_SIZE)), desc="Upserting"):
        batch = items[i : i + BATCH_SIZE]
        upsert_batch(client, batch)

    print(f"Ingestion completed. Total items: {len(items)}")
    return len(items)
