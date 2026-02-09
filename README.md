## eSchool Chatbot

This repository contains **one main production-style chatbot app (English)** plus two experimental copies:

- **Main app (`app/` + `run.py`)** – the primary version to focus on; this is the one intended for real use.
- **Arabic app (`app_arabic/` + `run_arabic.py`)** – a copy of the main app adapted for Arabic content and prompts.
- **Nomic app (`app_nomic/` + `run_nomic.py`)** – a copy of the main app used to test the Nomic embedding model.

An AI-powered retrieval‑augmented chatbot for eSchool website https://web.myeschoolhome.com/  
It uses sentence embeddings (EmbeddingGemma via `sentence-transformers`) and Qdrant as a vector database to answer user questions about eSchool by searching over a curated knowledge base.

---

## Features

- **RAG chatbot UI**: Web chat interface built with Flask (`/` and `/query` routes in `app/routes.py` and `app/templates/index.html`).
- **Context‑aware answers**:  
  - Classifies each user query as *related* or *unrelated* using DeepSeek V3 (via Ollama) and custom prompts.  
  - For related queries, retrieves relevant chunks from Qdrant and generates a grounded answer using DeepSeek V3.
- **Conversation history & token management**:  
  - Maintains `question`/`answer` history and truncates it based on a configurable token limit.
- **Content ingestion pipeline**:  
  - Script `create_qdrant_collection.py` builds/updates a Qdrant collection from `text_to_embed.json` using a local `SentenceTransformer` model (EmbeddingGemma).
- **Admin panel for content**:  
  - From the web UI, insert / view / update / delete chunks in Qdrant via REST endpoints (`/chunks` routes).

---

## Project Structure

- **Main app (English, production-like)**  
  - **`run.py`** – Entry point for the main Flask web app (binds to `PORT` from `.env`).
  - **`app/`** – Main application package:
    - **`__init__.py`** – Application factory (`create_app`) that:
      - Loads `.env` via `python-dotenv`.
      - Configures Flask from `config.py`.
      - Initializes:
        - `SentenceTransformer` embedder (`EMBEDDING_MODEL`).
        - `ollama.Client` (`OLLAMA_HOST`, `OLLAMA_API_KEY`, `MODEL_NAME`).
        - `QdrantClient` (`QDRANT_HTTP`).
      - Loads system and user prompts from `prompts/`.
      - Pre‑computes token count for the answer system prompt.
    - **`routes/`** – Flask blueprints / route modules:
      - Expose the chat UI (`GET /`).
      - Expose the main chat endpoint (`POST /query`).
      - Expose CRUD endpoints for Qdrant chunks (`/chunks`).
    - **`services/`** – Core business logic:
      - Query preprocessing and classification (related / unrelated to eSchool).
      - Retrieval of top‑K chunks from Qdrant.
      - Conversation history management and token‑budget truncation.
      - Answer generation with Ollama using prompts + retrieved context.
    - **`templates/index.html`** – Single‑page chat UI:
      - Chat stream with user/assistant messages.
      - “Clear history” button.
      - Admin panel to manage Qdrant chunks via the REST API.

- **Arabic app (experimental)**  
  - **`run_arabic.py`** – Entry point for the Arabic version.
  - **`app_arabic/`** – Copy of the main app adapted for Arabic:
    - Similar structure to `app/` (`__init__.py`, `arabic_routes/`, `arabic_services/`, `templates/`).
    - Uses Arabic prompts from `prompts/` (e.g. `arabic_answer_generation_system_prompt.txt`).
    - Can optionally use Arabic embedding models from `arabic_embedding_models/`.

- **Nomic embeddings app (experimental)**  
  - **`run_nomic.py`** – Entry point for the Nomic‑based experiment.
  - **`app_nomic/`** – Copy of the main app:
    - Same high‑level flow (classify → retrieve → generate answer).
    - Swaps the embedding model to Nomic for experimentation.

- **API‑based experiment (Gemini / other APIs)**  
  - **`aside/app_api.py`** – Alternative / experimental Flask app:
    - Uses API‑based embeddings (e.g. Gemini via `google.genai`) instead of the local `SentenceTransformer`.
    - Exposes its own `/query` and `/chunks` endpoints, separate from the main `app/` package.
    - Intended only for testing API embedding models; not part of the main deployment path.

- **Shared configuration and data**
  - **`config.py`** – Central configuration loaded from environment:
    - **Embedding / ingestion**
      - `EMBEDDING_MODEL`
      - `TEXT_TO_EMBED_PATH`
      - `VECTOR_SIZE`
      - `BATCH_SIZE`
      - `DISTANCE`
      - `NORMALIZE_EMBEDDINGS`
    - **Retrieval / chat**
      - `NUM_RETRIEVED_CHUNKS`
      - `TOKEN_LIMIT`
    - **Qdrant**
      - `COLLECTION_NAME`
      - `QDRANT_HTTP`
    - **Ollama**
      - `OLLAMA_API_KEY`
      - `MODEL_NAME`
      - `OLLAMA_HOST`
    - **Flask**
      - `PORT`
      - `FLASK_DEBUG`
    - **Blocking**
      - `BLOCK_THRESHOLD`
      - `BLOCK_MESSAGE`
  - **`create_qdrant_collection.py`** – Offline ingestion script:
    - Loads items from `TEXT_TO_EMBED_PATH` (`text_to_embed.json` by default).
    - Embeds `text` with `SentenceTransformer(EMBEDDING_MODEL)` and upserts to Qdrant collection `COLLECTION_NAME`.
  - **`text_to_embed.json`** / **`text_to_embed_arabic.json`** – Example knowledge bases:
    - List of objects `{ "id": ..., "text": "...", "metadata": { "page_title": ..., "url": ..., "section_title": ... } }`.
  - **`prompts/`** – Prompt templates:
    - `answer_generation_system_prompt.txt`
    - `answer_generation_user_prompt.txt`
    - `decision_making_system_prompt.txt`
    - `decision_making_user_prompt.txt`
    - `arabic_answer_generation_system_prompt.txt`
    - `arabic_decision_making_system_prompt.txt`
    - `translation_prompt.txt`
  - **`embeddinggemma-300m/`** – Local copy of Google’s EmbeddingGemma model for `sentence-transformers`.
  - **`arabic_embedding_models/`** – Local Hugging Face‑style folders for Arabic embedding experiments (used only by the Arabic app).
  - **`requirements.txt`** – Python dependencies.

---

## Prerequisites

- **Python**: 3.10+ recommended.
- **Qdrant**: Running instance (through Docker):
    docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.16.1
- **Ollama (server)** with the chosen `MODEL_NAME` pulled (deepseek-v3.1:671b).
- **Embedding model**: `EMBEDDING_MODEL` must point to the local `embeddinggemma-300m` directory or a compatible HF model ID.

---

## Installation

- **1. Clone the repository** 
  git clone https://github.com/your-org/eschool-chatbot-main.git
  cd eschool-chatbot-main

  - **2. Create and activate a virtual environment** 
  python -m venv .venv
  .venv\Scripts\activate   # Windows PowerShell
  
  - **3. Install dependencies**
  pip install -r requirements.txt


## Configuration

Create a `.env` file in the project root (same folder as `config.py`) and set:

# Embeddings / ingestion
EMBEDDING_MODEL=./embeddinggemma-300m          # or a HF model name
TEXT_TO_EMBED_PATH=./text_to_embed.json
VECTOR_SIZE=768
BATCH_SIZE=32
DISTANCE=cosine                                # or dot / euclid
NORMALIZE_EMBEDDINGS=true

# Retrieval / chat
NUM_RETRIEVED_CHUNKS=5
TOKEN_LIMIT=2048

# Qdrant
COLLECTION_NAME=eschool_docs
QDRANT_HTTP=http://localhost:6333

# Ollama
OLLAMA_API_KEY=your_ollama_api_key_here
OLLAMA_HOST=http://localhost:11434            # or remote Ollama endpoint
MODEL_NAME=your-ollama-model-name             # e.g. deepseek-r1:latest

# Flask
PORT=5000
FLASK_DEBUG=true

# Blocking / filters
BLOCK_THRESHOLD=3
BLOCK_MESSAGE=Your queries seem unrelated to eSchool. Please stay on topic.

# If you use `aside/app_api.py` with Gemini, also set:
GOOGLE_API_KEY=your_gemini_api_key_here

---

## Building the Vector Index

Before running the chatbot, ingest your knowledge base into Qdrant:

# From project root, with .venv activated and .env configured
python create_qdrant_collection.py. This will:

- Load all items from `TEXT_TO_EMBED_PATH` (default `text_to_embed.json`).
- Create (if necessary) and populate the Qdrant collection `COLLECTION_NAME`.

You can replace `text_to_embed.json` with your own file in the same format to customize the knowledge base.

---

## Running the Web Chatbot

- **1. Start the main Flask app** 
  python run.py
    The app will bind to `0.0.0.0` on `PORT` from `.env` (e.g. `5000`).

- **2. Open the UI**
  In your browser:
  http://localhost:5000/    # or whatever PORT you configured

  - **3. Use the chatbot**

  - Type questions about eSchool (products, features, contacts, etc.).
  - The model will:
    - Decide if the question is related to the domain.
    - If related, retrieve relevant text chunks from Qdrant and generate a grounded answer.
    - Maintain multi‑turn history, truncated by token budget.

- **4. Manage chunks from the UI**

  At the bottom of the page you’ll find **Qdrant controls** to:
  - Insert new chunks (text + optional `page_title`, `url`, `section_title`).
  - View and update an existing chunk by numeric ID.
  - Delete a chunk by ID.

---

## Alternative API Server (Optional)

The `aside/app_api.py` file exposes a similar API (including `/query` and `/chunks`), but:

- Uses **API-based embeddings** (for example, Gemini via `google.genai`) instead of the local `SentenceTransformer`.
- Manages its own Flask app and Qdrant integration.
- Listens on port `5001` by default.

Run it with:

python aside/app_api.py

Only use this if you specifically need to test an API-based embedding / LLM provider; the recommended path is still via `run.py` and the main `app/` package.

---

## Customizing Prompts & Behavior

- Edit the prompt templates in `prompts/` to adjust:
  - **Decision‑making** (when to call retrieval, how to respond to unrelated queries).
  - **Answer style** (tone, structure, level of detail).
- Tune key parameters via `.env`:
  - `NUM_RETRIEVED_CHUNKS` – more/less context per answer.
  - `TOKEN_LIMIT` – max context length sent to the LLM.
  - `BLOCK_THRESHOLD` / `BLOCK_MESSAGE` – how aggressively to block off‑topic use.

---

## License & Model Usage

- The **EmbeddingGemma** model under `embeddinggemma-300m/` is governed by Google’s Gemma license (see `embeddinggemma-300m/README.md`).
- Ensure you comply with:
  - Gemma / EmbeddingGemma terms.
  - Qdrant and Ollama licenses.
  - Any LLM or API provider policies (e.g. Gemini, if used via `GOOGLE_API_KEY`).