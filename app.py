from flask import Flask, Response, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os
from ollama import Client
from dotenv import load_dotenv

# -----------------------------
# Configuration
# -----------------------------
COLLECTION_NAME = "docs"
QDRANT_HTTP = "http://localhost:6333"
EMBEDDING_MODEL = r"C:\Users\ADMIN\Documents\GitHub\eschool-chatbot\embeddinggemma\embeddinggemma-300m"
TOP_K = 30  # number of results to retrieve
load_dotenv()

# -----------------------------
# Initialize
# -----------------------------
app = Flask(__name__)
model = SentenceTransformer(EMBEDDING_MODEL)
qdrant_client = QdrantClient(url=QDRANT_HTTP)
OLLAMA_KEY = os.getenv("OLLAMA_API_KEY")
MODEL_NAME = "deepseek-v3.1:671b"
ollama_client = Client(
                    host="https://ollama.com",
                    headers={'Authorization': 'Bearer ' + OLLAMA_KEY}
                )

# -----------------------------
# Retrieval function
# -----------------------------
def retrieve(query: str, top_k: int = TOP_K):
    query_vector = model.encode([query], convert_to_tensor=False)[0].tolist()
    print("Query vector shape:", len(query_vector), "Vector snippet:", query_vector[:5])

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )
    print("Raw results from Qdrant:", results)


    points = getattr(results, "points", results)

    retrieved_items = []
    for res in points:
        retrieved_items.append({
            "id": res.id,
            "text": res.payload.get("text", ""),
            "metadata": {
                "page_title": res.payload.get("page_title", ""),
                "url": res.payload.get("url", ""),
                "section_title": res.payload.get("section_title", "")
            }
        })

    
    return retrieved_items

def generate_with_deepseek(system_prompt: str, user_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    print("Messages sent to Ollama:", messages)

    try:
        response = ollama_client.chat(MODEL_NAME, messages=messages)
        print("Raw Ollama response:", response)

        if hasattr(response, "message") and hasattr(response.message, "content"):
            answer = response.message.content
        # Fallback for older formats or dict/list responses
        elif isinstance(response, list) and len(response) > 0 and hasattr(response[0], "content"):
            answer = response[0].content
        else:
            answer = str(response)

        print("Ollama answer:", answer)
        return answer

    except Exception as e:
        return f"⚠️ Error: {e}"

    
# -----------------------------
# Flask routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index_qdrant.html")

@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.json
        user_query = data.get("query", "")
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        results = retrieve(user_query, top_k=TOP_K)

        if not results:
            # No context to generate from
            return jsonify({
                "query": user_query,
                "answer": "I don't have enough information in the provided context to answer that.",
                "context_count": 0
            })

        # Step 3: Build prompt for instruction-tuned model
        # Include URL and section title along with text for each chunk
        context_block = "\n\n".join(
            f"Text: {r['text']}\nSection: {r['metadata'].get('section_title','')}\nURL: {r['metadata'].get('url','')}"
            for r in results
        )


        system_prompt = """
    You are a helpful, truthful, and concise company assistant. 
    Your role is to answer user questions about the company and its website 
    using ONLY the information provided in the Context. Never make up facts.

    Instructions:
    1. For general questions:
        - Respond with a numbered list (1, 2, 3...).  
        - Each item must be a single clear idea; combine with related ideas.  
        - Use short, simple, and self-contained sentences.

    2. For specific questions:
        - Respond in short, precise paragraphs.  
        - Include all factual fields exactly as given (email, phone number, address, contact instructions).  
        - Do not omit information, even if it appears in only one chunk.

    3. Rewrite all content in clear, simple, natural language.  
    4. Never reference or mention the Context in your answer.  
    5. URLs:
        - Include urls only if they directly support a fact you mention.
        - Place the URL in parentheses immediately after the fact; do not list irrelevant URLs at the end.
    6. Missing Information: 
        - If the necessary information is absent, reply exactly:
            "I don't have enough information in the provided context to answer that."
        - Then suggest one brief next step.
    """


        user_prompt = f"""
    Context:
    {context_block}

    Question:
    {user_query}

    Answer:
    """
        
        # Step 3: Call Deepseek

        # Ollama chat returns a list of message objects; get content of first
        answer = generate_with_deepseek(system_prompt, user_prompt)

        # Step 4: Return JSON response
        return jsonify({
            "query": user_query,
            "answer": answer,
            "context_count": len(results),
            "context_results": results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
