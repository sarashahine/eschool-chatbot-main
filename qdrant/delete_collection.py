
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1️⃣ Connect to local Qdrant
# -----------------------------
client = QdrantClient(url="http://localhost:6333")

collection_name = "docs_api"


# Delete old collection
if client.get_collection(collection_name) is not None:
    client.delete_collection(collection_name)

# Create new collection with correct vector size
client.create_collection(
    collection_name=collection_name,
    vectors_config=rest.VectorParams(size=768, distance=rest.Distance.COSINE)
)
