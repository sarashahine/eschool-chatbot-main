from flask import Flask
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from ollama import Client
from qdrant_client import QdrantClient

from .routes import main_routes
from .utils import load_prompt
from config import QDRANT_HTTP, EMBEDDING_MODEL, OLLAMA_KEY

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config.from_object("config")

    if not OLLAMA_KEY:
        raise ValueError("OLLAMA_API_KEY is missing in .env")
    app.ollama_client = Client(
        host="https://ollama.com",
        headers={'Authorization': 'Bearer ' + OLLAMA_KEY}
    )

    app.qdrant_client = QdrantClient(url=QDRANT_HTTP)

    app.embedder = SentenceTransformer(EMBEDDING_MODEL)

    app.system_prompt = load_prompt("prompts/answer_generation_system_prompt.txt")
    app.preprocess_prompt = load_prompt("prompts/decision_making_system_prompt.txt")

    # Register routes
    app.register_blueprint(main_routes)

    return app
