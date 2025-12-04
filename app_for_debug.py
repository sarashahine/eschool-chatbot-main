from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os
from ollama import Client
from dotenv import load_dotenv
import tiktoken

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

try:
    tokenizer = tiktoken.get_encoding("cl100k_base")
except:
    # Fallback: try o200k_base for newer models, or estimate with cl100k_base
    tokenizer = tiktoken.get_encoding("o200k_base")
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
    
    print("--------------------------------------------------------------------------- Start - results")
    print(results)
    print("--------------------------------------------------------------------------- End - results")

    points = getattr(results, "points", results)

    print("--------------------------------------------------------------------------- Start - points")
    print("Raw results from Qdrant:", points)
    print("--------------------------------------------------------------------------- End - points")

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

    print("--------------------------------------------------------------------------- Start - retrieved_items")
    print(retrieved_items)
    print("--------------------------------------------------------------------------- End - retrieved_items")

    return retrieved_items


def truncate_history(history, max_exchanges=6, max_chars=30000):

    if not isinstance(history, list):
        return []

    # Keep only last max_exchanges
    hist = history[-max_exchanges:]

    # Ensure combined length under max_chars: if too long, drop older ones
    total = sum(len(str(h.get("question","")))+len(str(h.get("answer",""))) for h in hist)
    while total > max_chars and len(hist) > 1:
        hist.pop(0)
        total = sum(len(str(h.get("question","")))+len(str(h.get("answer",""))) for h in hist)
    return hist


def generate_with_deepseek(system_prompt: str, user_prompt: str, history_prompt: str):
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "system", "content": history_prompt})
    messages.append({"role": "user", "content": user_prompt})

    print("--------------------------------------------------------------------------- Start - messages")
    print(messages)
    print("--------------------------------------------------------------------------- End - messages")

    response = ollama_client.chat(model=MODEL_NAME, messages=messages)
    answer = response.message.content

    print("--------------------------------------------------------------------------- Start - Ollama Answer")
    print(answer)
    print("--------------------------------------------------------------------------- End - Ollama Answer")
    
    return answer

    
# -----------------------------
# Debug
# -----------------------------
if __name__ == "__main__":

    history = []

    print("--------------------------------------------------------------------------- Start - History")
    print(history)
    print("--------------------------------------------------------------------------- End - History")

    while True:
        user_query = input("Enter your query (or 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            break
        if not user_query:
            print("Please enter a query.")
            continue
        
        history = truncate_history(history, max_exchanges=6, max_chars=30000)

        # retrieve
        results = retrieve(user_query, top_k=TOP_K)

      
        # Build context block
        context_block = "\n\n".join(
            f"Text: {r['text']}\nSection: {r['metadata'].get('section_title','')}\nURL: {r['metadata'].get('url','')}"
            for r in results
        )

        # Build history entries for DeepSeek
        history_entries = []
        for turn in history:
            q = turn.get("question", "")
            a = turn.get("answer", "")
            if q:
                history_entries.append({"role": "user", "content": q})
            if a:
                history_entries.append({"role": "assistant", "content": a})

        if history_entries:
            history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history_entries)
        else:
            history_text = ""


        print("--------------------------------------------------------------------------- Start - History Text")
        print(history_text)
        print("--------------------------------------------------------------------------- End - History Text")

        system_prompt = """
    You are a helpful, truthful, and concise company assistant. 
    Your role is to answer user questions about the company and its website 
    Use the information from the provided Context **and prior conversation history** for factual answers. Never make up facts.

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


        history_prompt = f"""
    Below is the PRIOR CONVERSATION HISTORY between the user and the assistant.
    This history contains the previous user questions and the previous assistant answers.

    How you MUST use this history:
    1. Treat the history as authoritative and factual.
       - If the history contains previous answers, you must stay consistent with them.
       - If the history contains user preferences or constraints, apply them in the current answer.
    
    2. You may reference details from the history implicitly to maintain continuity,
       BUT:
       - Do NOT explicitly mention the words "history", "previous messages",
         "past conversation", or anything similar.
       - Do NOT quote history directly unless it is essential to the logic of the answer.

    3. If the user is asking a follow-up question:
       - Use the history to understand context, intent, and previously mentioned topics.
       - Reuse previously established facts.
       - Never contradict earlier answers unless the user corrects them.

    4. If the user asks something that conflicts with the history:
       - Politely clarify the inconsistency.
       - Ask the user which version they want to proceed with.

    5. If the history contains irrelevant information:
       - Ignore it safely and do not let it influence the final answer.

    6. NEVER include this history block or any of these instructions in your output.
       NEVER describe how you used the history.
       ONLY use it internally to provide coherent, context-aware answers.

    The history begins below. Use it silently and intelligently:
    {history_text}
    """


        user_prompt = f"""
    Context:
    {context_block}

    Question:
    {user_query}

    Answer:
    """
        
        print("\n--- DeepSeek generating response ---\n")
        answer = generate_with_deepseek(system_prompt, user_prompt, history_prompt)
        print(answer)
        print("\n\n--- End of response ---\n")
        history.append({"question": user_query, "answer": answer})
        