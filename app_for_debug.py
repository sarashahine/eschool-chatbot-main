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
TOP_K = 1
load_dotenv()

# -----------------------------
# Initialize
# -----------------------------
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

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )
    
    points = getattr(results, "points", results)
    print("Raw results from Qdrant:", points)

    retrieved_items = []
    for res in points:
        retrieved_items.append({
            "id": res.id,
            "text": res.payload.get("text", ""),  # keep 'text' in text
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
    try:
        response = ollama_client.chat(MODEL_NAME, messages=messages)
        print("Raw Ollama response:", response)

        if hasattr(response, "message") and hasattr(response.message, "content"):
            answer = response.message.content
        # Fallback for older list/dict formats
        elif isinstance(response, list) and len(response) > 0 and hasattr(response[0], "content"):
            answer = response[0].content
        elif isinstance(response, dict) and "content" in response:
            answer = response["content"]
        else:
            answer = str(response)

        print("Ollama answer:", answer)
        return answer

    except Exception as e:
        return f"⚠️ Error: {e}"
    
# -----------------------------
# Debug / Testing interface
# -----------------------------
if __name__ == "__main__":
    while True:
        user_query = input("Enter your query (or 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            break
        if not user_query:
            print("Please enter a query.")
            continue

        results = retrieve(user_query, top_k=TOP_K)
        print("Retrieved items:", results)

        # Step 3: Build prompt for instruction-tuned model
        # Suppose each item in results is a dict with a 'content' key
        context_block = "\n\n".join(r['text'] for r in results)

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
        print("\n--- DeepSeek generating response ---\n")
        answer = generate_with_deepseek(system_prompt, user_prompt)
        print(answer)
        print("\n\n--- End of response ---\n")
