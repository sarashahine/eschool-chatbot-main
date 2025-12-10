import json
import os
import re
from dotenv import load_dotenv
from ollama import Client
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config import COLLECTION_NAME, QDRANT_HTTP, EMBEDDING_MODEL, TOP_K, MODEL_NAME, TOKEN_LIMIT

load_dotenv()


# -----------------------------
# Initialize
# -----------------------------
OLLAMA_KEY = os.getenv("OLLAMA_API_KEY")
if not OLLAMA_KEY:
    raise ValueError("OLLAMA_API_KEY is missing in .env")

model = SentenceTransformer(EMBEDDING_MODEL)
ollama_client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + OLLAMA_KEY}
)
qdrant_client = QdrantClient(url=QDRANT_HTTP)


# -----------------------------
# Load prompts from external files
# -----------------------------
try:
    with open("prompts/system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT_BASE = f.read()
        print("SYSTEM_PROMPT_BASE read successfully")
except Exception:
    SYSTEM_PROMPT_BASE = ""
    print("Warning: 'prompts/system_prompt.txt' not found. Using empty system prompt.")

try:
    with open("prompts/history_prompt.txt", "r", encoding="utf-8") as f:
        HISTORY_PROMPT_BASE = f.read()
        print("HISTORY_PROMPT_BASE read successfully")
except Exception:
    HISTORY_PROMPT_BASE = ""
    print("Warning: 'prompts/history_prompt.txt' not found. Using empty history prompt.")

try:
    with open("prompts/preprocess_prompt.txt", "r", encoding="utf-8") as f:
        PREPROCESS_PROMPT_BASE = f.read()
        print("PREPROCESS_PROMPT_BASE read successfully")
except Exception:
    PREPROCESS_PROMPT_BASE = ""
    print("Warning: 'prompts/preprocess_prompt.txt' not found. Using empty history prompt.")

# -----------------------------
# Helper Functions
# -----------------------------
def retrieve(query: str, top_k: int = TOP_K):
    print("Retrieving for query:", query)
    query_vector = model.encode([query], convert_to_tensor=False)[0].tolist()
    print("Query vector shape/first 5 elements:", query_vector[:5])

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )

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

    with open('retrieved_items.json', 'w', encoding='utf-8') as f:
        json.dump(retrieved_items, f, indent=4, ensure_ascii=False)
    print("JSON saved to 'retrieved_items.json' in a readable format.")

    return retrieved_items


def safe_json_load(llm_output: str):
    cleaned = re.sub(r"^```json\s*|```$", "", llm_output.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print("Failed to parse JSON. Cleaned output:", repr(cleaned))
        return {"category": "general", "direct_answer": "No valid JSON received."}
    
def pre_process_query(user_query, history=None, log_file2="reformulator_log.txt"):
    messages = []
    
    if history:
        for turn in history:
            q = turn.get("question", "").strip()
            a = turn.get("answer", "").strip()
            if q:
                messages.append({"role": "user", "content": q})
            if a:
                messages.append({"role": "assistant", "content": a})
    
    messages.append({"role": "user", "content": user_query})

    try:
        messages=[{"role": "system", "content": PREPROCESS_PROMPT_BASE}, *messages]
        response = ollama_client.chat(model=MODEL_NAME, messages=messages)
        llm_output = response.message.content
        print("Raw llm_output:", repr(llm_output))

        result = safe_json_load(llm_output)
        
        print("result: ", result)
        category = result.get("category")
        direct_answer = result.get("direct_answer")
        retrieval_query = result.get("retrieval_query")

        requires_retrieval = category == "retrieval"
        print("requires_retrieval: ", requires_retrieval)


        separator = "\n\n\n" + "="*100 + "\n\n\n\n"
        if not history:
            if os.path.exists(log_file2):
                with open(log_file2, "w", encoding="utf-8") as f:
                    f.write("")
                print(f"History empty. Cleared '{log_file2}'")
        with open(log_file2, "a", encoding="utf-8") as f:
            # f.write("Prompt Sent to LLM:\n")
            # f.write(llm_prompt_text + "\n\n")
            f.write("Messages sent:\n")
            f.write(json.dumps(messages, indent=2, ensure_ascii=False) + "\n")
            f.write("LLM Output:\n")
            f.write(json.dumps(llm_output, indent=2, ensure_ascii=False) + "\n")
            f.write("Cleaned LLM Output:\n")
            f.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            f.write("Requires retrieval: ")
            f.write(json.dumps(requires_retrieval, indent=2, ensure_ascii=False) + "\n")
            f.write("Retrieval query: ")
            f.write(json.dumps(retrieval_query, indent=2, ensure_ascii=False) + "\n")
            f.write(separator)



        return {
            "requires_retrieval": requires_retrieval,
            "direct_answer": direct_answer,
            "retrieval_query": retrieval_query if requires_retrieval else None
        }
    except Exception as e:
        print("Error in pre_process_query:", e)
        # fallback: assume retrieval required
        return {
            "requires_retrieval": True,
            "direct_answer": None,
            "retrieval_query": user_query
        }



def count_tokens(text):
    try:
        result = ollama_client.tokens(model=MODEL_NAME, prompt=text)
        return result.get("total_tokens", len(text.split()))
    except Exception:
        return len(text.split())


def truncate_history(history, user_query, context_block, token_limit=TOKEN_LIMIT):

    system_tokens = count_tokens(SYSTEM_PROMPT_BASE)
    history_tokens = count_tokens(HISTORY_PROMPT_BASE)
    user_tokens = count_tokens(user_query)
    context_tokens = count_tokens(context_block)

    total_tokens = system_tokens + history_tokens + user_tokens + context_tokens

    if not isinstance(history, list):
        history = []

    hist = history[::-1]
    truncated_history = []

    for h in hist:
        q = h.get("question", "")
        a = h.get("answer", "")
        h_tokens = count_tokens(q) + count_tokens(a)

        if total_tokens + h_tokens <= token_limit:
            truncated_history.append(h)
            total_tokens += h_tokens
        else:
            break

    return truncated_history[::-1]




def log_llm_interaction(llm_prompt_text, llm_answer, history=None, log_file="llm_interaction_log.txt"):
    separator = "\n\n\n" + "=" * 100 + "\n\n\n\n"
    if not history:
        if os.path.exists(log_file):
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("")
            print(f"History empty. Cleared '{log_file}'")

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("Prompt Sent to LLM:\n")
        f.write(llm_prompt_text + "\n\n")
        f.write("LLM Answer:\n")
        f.write(llm_answer + "\n")
        f.write(separator)
    print(f"Interaction logged to '{log_file}'")


def generate_with_deepseek(system_prompt: str, user_prompt: str, history_prompt: str, history_turns: list):
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "system", "content": history_prompt})
    for turn in history_turns:
        messages.append(turn)
    messages.append({"role": "user", "content": user_prompt})


    try:
        response = ollama_client.chat(model=MODEL_NAME, messages=messages)
        answer = response.message.content
    except Exception as e:
        print("Error during LLM call:", e)
        raise e

    print("--------------------------------------------------------------------------- Start - Ollama Answer")
    print(answer)
    print("--------------------------------------------------------------------------- End - Ollama Answer")

    full_prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    log_llm_interaction(llm_prompt_text=full_prompt_text, llm_answer=answer, history=history_turns)
    
    return answer


# -----------------------------
# Debug CLI
# -----------------------------
if __name__ == "__main__":
    history = []

    while True:
        user_query = input("Enter your query (or 'exit' to quit): ").strip()
        if user_query.lower() == "exit":
            break
        if not user_query:
            print("Please enter a query.")
            continue

        # Build history entries before routing
        history_entries = []
        for turn in history:
            q = turn.get("question", "")
            a = turn.get("answer", "")
            if q:
                history_entries.append({"role": "user", "content": q})
            if a:
                history_entries.append({"role": "assistant", "content": a})

        decision = pre_process_query(user_query, history)
        if not decision["requires_retrieval"]:
            print("answer: " + decision["direct_answer"])
            continue

        retrieval_query = decision["retrieval_query"]
        print("retrieval query: ", retrieval_query)
        results = retrieve(retrieval_query, top_k=TOP_K)
        # results = retrieve(user_query, top_k=TOP_K)

        context_block = "\n\n".join(
            f"Text: {r['text']}\nSection: {r['metadata'].get('section_title','')}\nURL: {r['metadata'].get('url','')}"
            for r in results
        )

        history = truncate_history(history, user_query, context_block)

        # Rebuild history entries from truncated history for generation
        history_entries = []
        for turn in history:
            q = turn.get("question", "")
            a = turn.get("answer", "")
            if q:
                history_entries.append({"role": "user", "content": q})
            if a:
                history_entries.append({"role": "assistant", "content": a})

        system_prompt = SYSTEM_PROMPT_BASE
        history_prompt = HISTORY_PROMPT_BASE
        user_prompt = f"""
    Context:
    {context_block}

    Question:
    {user_query}

    Answer:
    """
        
        print("\n--- DeepSeek generating response ---\n")
        answer = generate_with_deepseek(system_prompt, user_prompt, history_prompt, history_entries)
        print("\n--- End of response ---\n")

        history.append({"question": user_query, "answer": answer})