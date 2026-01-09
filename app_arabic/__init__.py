from flask import Flask
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from ollama import Client
from qdrant_client import QdrantClient

from .arabic_routes.arabic_main_routes import arabic_main_routes
from .arabic_routes.arabic_chunk_routes import arabic_chunk_routes
from .arabic_routes.arabic_admin_routes import arabic_admin_routes
from .arabic_services.utils import load_prompt
from .arabic_services.arabic_retrieval import count_tokens
from config import QDRANT_HTTP, ARABIC_EMBEDDING_MODEL, OLLAMA_KEY, OLLAMA_HOST

load_dotenv()

def create_app():
    arabic_app = Flask(__name__)
    arabic_app.config.from_object("config")

    if not OLLAMA_KEY:
        raise ValueError("OLLAMA_API_KEY is missing in .env")
    arabic_app.ollama_client = Client(
        host=OLLAMA_HOST,
        headers={'Authorization': f'Bearer ' + OLLAMA_KEY}
    )

    arabic_app.qdrant_client = QdrantClient(url=QDRANT_HTTP)

    arabic_app.embedder = SentenceTransformer(ARABIC_EMBEDDING_MODEL)

    arabic_app.answer_generation_system_prompt = load_prompt("prompts/arabic_answer_generation_system_prompt.txt")
    arabic_app.answer_generation_user_prompt = load_prompt("prompts/answer_generation_user_prompt.txt")

    arabic_app.decision_making_system_prompt = load_prompt("prompts/arabic_decision_making_system_prompt.txt")
    arabic_app.decision_making_user_prompt = load_prompt("prompts/decision_making_user_prompt.txt")

    arabic_app.answer_generation_system_prompt_tokens = count_tokens(arabic_app.answer_generation_system_prompt, arabic_app.ollama_client)

    arabic_app.translation_prompt = load_prompt("prompts/translation_prompt.txt")

    # Register routes
    arabic_app.register_blueprint(arabic_main_routes)
    arabic_app.register_blueprint(arabic_chunk_routes)
    arabic_app.register_blueprint(arabic_admin_routes, url_prefix="/admin")

    return arabic_app
