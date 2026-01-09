import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# -----------------------------
# Embedding settings
# -----------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
TEXT_TO_EMBED_PATH=os.getenv("TEXT_TO_EMBED_PATH")
ARABIC_TEXT_TO_EMBED_PATH = os.getenv("ARABIC_TEXT_TO_EMBED_PATH")
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
OLLAMA_LOCAL_HOST = os.getenv("OLLAMA_LOCAL_HOST")
HISTORY_START_MESSAGE = os.getenv("HISTORY_START_MESSAGE")
HISTORY_END_MESSAGE = os.getenv("HISTORY_END_MESSAGE")
RETRIEVAL_START_MESSAGE=os.getenv("RETRIEVAL_START_MESSAGE")
RETRIEVAL_END_MESSAGE=os.getenv("RETRIEVAL_END_MESSAGE")

# -----------------------------
# Flask settings
# -----------------------------
PORT = os.getenv("PORT")
NOMIC_PORT = os.getenv("NOMIC_PORT")
ARABIC_PORT = os.getenv("ARABIC_PORT")
FLASK_DEBUG = os.getenv("FLASK_DEBUG")

# -----------------------------
# Logging settings
# -----------------------------
LOG_MAX_LINES = int(os.getenv("LOG_MAX_LINES"))
LOG_DIR = os.getenv("LOG_DIR")
LOG_FILE_BASENAME = os.getenv("LOG_FILE_BASENAME")
LOG_FILE_EXT = os.getenv("LOG_FILE_EXT")

# -----------------------------
# Blocking / Filters
# -----------------------------
BLOCK_THRESHOLD = int(os.getenv("BLOCK_THRESHOLD"))
BLOCK_MESSAGE = os.getenv("BLOCK_MESSAGE")

# -----------------------------
# Nomic Embedding settings
# -----------------------------
NOMIC_EMBED_URL = os.getenv("NOMIC_EMBED_URL")
NOMIC_MODEL_NAME = os.getenv("NOMIC_MODEL_NAME")
NOMIC_COLLECTION_NAME = os.getenv("NOMIC_COLLECTION_NAME")

# -----------------------------
# Arabic settings
# -----------------------------
ARABIC_EMBEDDING_MODEL = os.getenv("ARABIC_EMBEDDING_MODEL")
ARABIC_COLLECTION_NAME = os.getenv("ARABIC_COLLECTION_NAME")
ARABIC_MODEL_NAME = os.getenv("ARABIC_MODEL_NAME")