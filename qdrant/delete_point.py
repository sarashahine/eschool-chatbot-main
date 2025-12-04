from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
COLLECTION_NAME = "docs"
QDRANT_HTTP = "http://localhost:6333"
EMBEDDING_MODEL = r"C:\Users\ADMIN\Documents\GitHub\eschool-chatbot\embeddinggemma\embeddinggemma-300m"

# -----------------------------
# Init client + embed model
# -----------------------------
client = QdrantClient(url=QDRANT_HTTP)
model = SentenceTransformer(EMBEDDING_MODEL)

def embed(text):
    return model.encode([text])[0].tolist()

# -----------------------------
# 1. DELETE TEST
# -----------------------------
TEST_DELETE_ID = 23   # choose any existing ID in your DB

print("\n=== DELETE TEST ===")
print(f"Deleting point ID = {TEST_DELETE_ID}")

client.delete(
    collection_name=COLLECTION_NAME,
    points_selector=rest.PointIdsList(points=[TEST_DELETE_ID]),
    wait=True
)

# Verify deletion
res = client.retrieve(
    collection_name=COLLECTION_NAME,
    ids=[TEST_DELETE_ID]
)

print("Retrieve after delete:", res)   # should be empty list