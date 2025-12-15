import os

COLLECTION_NAME = "docs"
QDRANT_HTTP = "http://localhost:6333"
EMBEDDING_MODEL = r"C:\Users\ADMIN\Documents\GitHub\eschool-chatbot-main\embeddinggemma-300m"
TOP_K = 15  # number of results to retrieve
OLLAMA_KEY = os.getenv("OLLAMA_API_KEY")
MODEL_NAME = "deepseek-v3.1:671b"
VECTOR_SIZE = 768
TOKEN_LIMIT = 128000
BLOCK_THRESHOLD = 2000
BLOCK_MESSAGE = "For more info visit our website https://web.myeschoolhome.com/"