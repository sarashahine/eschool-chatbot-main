import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# -----------------------------
# Embedding settings
# -----------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
TEXT_TO_EMBED_PATH=os.getenv("TEXT_TO_EMBED_PATH")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE"))
BATCH_SIZE=int(os.getenv("BATCH_SIZE"))
DISTANCE=os.getenv("DISTANCE")
NORMALIZE_EMBEDDINGS=os.getenv("NORMALIZE_EMBEDDINGS")

# -----------------------------
# Retrieval / Chat settings
# -----------------------------
NUM_RETRIEVED_CHUNKS = int(os.getenv("NUM_RETRIEVED_CHUNKS"))
TOKEN_LIMIT = int(os.getenv("TOKEN_LIMIT"))

# -----------------------------
# Qdrant settings
# -----------------------------
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
QDRANT_HTTP = os.getenv("QDRANT_HTTP")

# -----------------------------
# Ollama settings
# -----------------------------
OLLAMA_KEY = os.getenv("OLLAMA_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
OLLAMA_HOST=os.getenv("OLLAMA_HOST")

# -----------------------------
# Flask settings
# -----------------------------
PORT = os.getenv("PORT")
FLASK_DEBUG = os.getenv("FLASK_DEBUG")

# -----------------------------
# Logging settings
# -----------------------------
LOG_MAX_LINES = int(os.getenv("LOG_MAX_LINES"))
LOGGING_FILE = os.getenv("LOGGING_FILE")

# -----------------------------
# Blocking / Filters
# -----------------------------
BLOCK_THRESHOLD = int(os.getenv("BLOCK_THRESHOLD"))
BLOCK_MESSAGE = os.getenv("BLOCK_MESSAGE")
