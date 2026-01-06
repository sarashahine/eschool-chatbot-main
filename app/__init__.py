from flask import Flask
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from ollama import Client
from qdrant_client import QdrantClient

from .routes.main_routes import main_routes
from .routes.chunk_routes import chunk_routes
from .routes.admin_routes import admin_routes
from .services.utils import load_prompt
from .services.retrieval import count_tokens
from config import QDRANT_HTTP, EMBEDDING_MODEL, OLLAMA_KEY, OLLAMA_HOST

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config.from_object("config")

    if not OLLAMA_KEY:
        raise ValueError("OLLAMA_API_KEY is missing in .env")
    app.ollama_client = Client(
        host=OLLAMA_HOST,
        headers={'Authorization': f'Bearer ' + OLLAMA_KEY}
    )

    app.qdrant_client = QdrantClient(url=QDRANT_HTTP)

    app.embedder = SentenceTransformer(EMBEDDING_MODEL)

    app.answer_generation_system_prompt = load_prompt("prompts/answer_generation_system_prompt.txt")
    app.answer_generation_user_prompt = load_prompt("prompts/answer_generation_user_prompt.txt")

    app.decision_making_system_prompt = load_prompt("prompts/decision_making_system_prompt.txt")
    app.decision_making_user_prompt = load_prompt("prompts/decision_making_user_prompt.txt")

    app.answer_generation_system_prompt_tokens = count_tokens(app.answer_generation_system_prompt, app.ollama_client)

    app.translation_prompt = load_prompt("prompts/translation_prompt.txt")

    # Register routes
    app.register_blueprint(main_routes)
    app.register_blueprint(chunk_routes)
    app.register_blueprint(admin_routes, url_prefix="/admin")

    return app
