import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# -----------------------------
# General settings
# -----------------------------
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
QDRANT_HTTP = os.getenv("QDRANT_HTTP")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE"))

# -----------------------------
# Ollama settings
# -----------------------------
OLLAMA_KEY = os.getenv("OLLAMA_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# -----------------------------
# Retrieval / Chat settings
# -----------------------------
NUM_RETRIEVED_CHUNKS = int(os.getenv("NUM_RETRIEVED_CHUNKS"))
TOKEN_LIMIT = int(os.getenv("TOKEN_LIMIT"))

# -----------------------------
# Blocking / Filters
# -----------------------------
BLOCK_THRESHOLD = int(os.getenv("BLOCK_THRESHOLD"))
BLOCK_MESSAGE = os.getenv("BLOCK_MESSAGE")

PORT = os.getenv("PORT")
FLASK_DEBUG = os.getenv("FLASK_DEBUG")
TEXT_TO_EMBED_PATH=os.getenv("TEXT_TO_EMBED_PATH")
BATCH_SIZE=int(os.getenv("BATCH_SIZE"))
DISTANCE=os.getenv("DISTANCE")
NORMALIZE_EMBEDDINGS=os.getenv("NORMALIZE_EMBEDDINGS")