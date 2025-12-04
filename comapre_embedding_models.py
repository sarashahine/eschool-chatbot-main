import os
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()


# ---------- GEMINI SETUP ----------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai_client = genai.Client(api_key=GOOGLE_API_KEY)

GEMINI_COLLECTION_NAME = "docs_api"
GEMINI_VECTOR_SIZE = 768

qdrant_gemini = QdrantClient(url="http://localhost:6333")

# ---------- GEMMA SETUP ----------
GEMMA_MODEL_PATH = r"C:\Users\ADMIN\Documents\GitHub\eschool-chatbot\embeddinggemma\embeddinggemma-300m"
gemma_model = SentenceTransformer(GEMMA_MODEL_PATH)

GEMMA_COLLECTION_NAME = "docs"
GEMMA_VECTOR_SIZE = 768

qdrant_gemma = QdrantClient(url="http://localhost:6333")

# ---------- Helper Functions ----------

def _normalize(vec):
    vec = np.array(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return (vec / norm).tolist()

# ---------- GEMINI FUNCTIONS ----------

def _embed_query_gemini(query: str):
    result = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query],
        config=types.EmbedContentConfig(output_dimensionality=GEMINI_VECTOR_SIZE),
    )
    return result.embeddings[0].values

def search_gemini(query: str, top_k: int = 10):
    qvec = _normalize(_embed_query_gemini(query))

    res = qdrant_gemini.query_points(
        collection_name=GEMINI_COLLECTION_NAME,
        query=qvec,
        limit=top_k,
    )

    points = getattr(res, "points", res)

    results = []
    for r in points:
        results.append({
            "id": r.id,
            "score": r.score,
            "text": r.payload.get("text", ""),
            "page_title": r.payload.get("page_title", ""),
            "section_title": r.payload.get("section_title", ""),
            "url": r.payload.get("url", ""),
        })

    return results

# ---------- GEMMA FUNCTIONS ----------
def _embed_query_gemma(query: str):
    vec = gemma_model.encode([query], convert_to_numpy=True)[0]
    vec = np.array(vec, dtype=float)
    norm = np.linalg.norm(vec)
    if norm != 0:
        vec = vec / norm
    return vec.tolist()


def search_gemma(query: str, top_k: int = 10):
    """
    Run semantic search against the Gemma-based Qdrant collection.
    Returns a list of dicts: {id, score, text, page_title, section_title, url}
    """
    qvec = _normalize(_embed_query_gemma(query))

    res = qdrant_gemma.query_points(
        collection_name=GEMMA_COLLECTION_NAME,
        query=qvec,
        limit=top_k,
    )

    points = getattr(res, "points", res)

    results = []
    for r in points:
        results.append({
            "id": r.id,
            "score": r.score,
            "text": r.payload.get("text", ""),
            "page_title": r.payload.get("page_title", ""),
            "section_title": r.payload.get("section_title", ""),
            "url": r.payload.get("url", ""),
        })

    return results


# Pseudo-code to run inside a notebook where you can import both search functions

eval_questions = [
    "How can parents view their child's attendance?",
    "How do teachers upload homework?",
    "How do I manage students grades?",
    "What reports can administrators generate?",
]

def pretty_print_results(model_name, results, top_k=3):
    print(f"\n=== {model_name} ===")
    for i, r in enumerate(results[:top_k], 1):
        print(f"\nResult {i} | score={r['score']:.3f}")
        print(r["text"])

for q in eval_questions:
    print("\n" + "#" * 80)
    print("QUERY:", q)

    gemini_res = search_gemini(q, top_k=5)  # from gemini notebook
    gemma_res = search_gemma(q, top_k=5)    # from gemma notebook

    pretty_print_results("GEMINI", gemini_res)
    pretty_print_results("GEMMA", gemma_res)