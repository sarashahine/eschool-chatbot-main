import os
from ollama import Client
from qdrant_client import QdrantClient

COLLECTION_NAME = "docs"
QDRANT_HTTP = "http://localhost:6333"
EMBEDDING_MODEL = r"C:\Users\ADMIN\Documents\GitHub\eschool-chatbot-main\embeddinggemma-300m"
TOP_K = 15  # number of results to retrieve
OLLAMA_KEY = os.getenv("OLLAMA_API_KEY")
MODEL_NAME = "deepseek-v3.1:671b"
VECTOR_SIZE = 768
TOKEN_LIMIT = 128000