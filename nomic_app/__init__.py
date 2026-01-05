from flask import Flask
from dotenv import load_dotenv
from ollama import Client
from qdrant_client import QdrantClient

from .nomic_routes.nomic_main_routes import nomic_main_routes
from .nomic_routes.nomic_chunk_routes import nomic_chunk_routes
from .nomic_routes.nomic_admin_routes import nomic_admin_routes
from .nomic_services.utils import load_prompt
from .nomic_services.nomic_retrieval import count_tokens
from config import QDRANT_HTTP, OLLAMA_KEY, OLLAMA_HOST

load_dotenv()

def create_app():
    nomic_app = Flask(__name__)
    nomic_app.config.from_object("config")

    if not OLLAMA_KEY:
        raise ValueError("OLLAMA_API_KEY is missing in .env")
    nomic_app.ollama_client = Client(
        host=OLLAMA_HOST,
        headers={'Authorization': f'Bearer ' + OLLAMA_KEY}
    )

    nomic_app.qdrant_client = QdrantClient(url=QDRANT_HTTP)

    nomic_app.answer_generation_system_prompt = load_prompt("prompts/answer_generation_system_prompt.txt")
    nomic_app.answer_generation_user_prompt = load_prompt("prompts/answer_generation_user_prompt.txt")

    nomic_app.decision_making_system_prompt = load_prompt("prompts/decision_making_system_prompt.txt")
    nomic_app.decision_making_user_prompt = load_prompt("prompts/decision_making_user_prompt.txt")

    nomic_app.answer_generation_system_prompt_tokens = count_tokens(nomic_app.answer_generation_system_prompt, nomic_app.ollama_client)

    # Register routes
    nomic_app.register_blueprint(nomic_main_routes)
    nomic_app.register_blueprint(nomic_chunk_routes)
    nomic_app.register_blueprint(nomic_admin_routes, url_prefix="/admin")

    return nomic_app
