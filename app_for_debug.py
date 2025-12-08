import json
import os
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


def pre_process_query(user_query, history=None):
    messages = []
    system_prompt = """
        You are an assistant for eSchool. Your job is to classify how the chatbot should handle the user's query.

        You MUST analyze the user's message **together with prior conversation history** and determine whether retrieval is needed.

        Your response must be ONLY valid JSON with:
        - "category": one of ["general", "unrelated", "retrieval"]
        - "retrieval_query": string or null
        - "direct_answer": string or null

        Rules:
        1. If the user is chit-chatting (hello, thanks, okay, etc.) → category="general" and provide a direct response.
        2. If the question has nothing to do with eSchool → category="unrelated" and provide a direct response: "I don't have an answer for this."
        3. If the question is about eSchool’s products, features, services, usage, roles, or any functional detail → category="retrieval".

        IMPORTANT — CORRECTIONS:
        If the user message is a correction, refinement, clarification, or continuation of a previous question 
        (e.g., “sorry I mean…”, “not student, teacher”, “I meant…”, “what about teachers”, “and for parents?”),
        you MUST:
        - treat it as if the user repeated the full question with the corrected part
        - set category="retrieval"
        - build a proper “retrieval_query” that includes the corrected meaning
        Example: 
        User: "how do I benefit from administrator as a student"
        Then user: "sorry I mean as a teacher"
        → retrieval_query must become something like:
        "How do teachers benefit from Administrator?"

        NEVER return "general" or "unrelated" when a correction is detected, even if the message alone is short or unclear.

        Format output as JSON only.
        """

    
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
        response = ollama_client.chat(model=MODEL_NAME, messages=[{"role": "system", "content": system_prompt}, *messages])
        llm_output = response.message.content
        print("llm_output: ", llm_output)
        # Try to parse JSON safely
        try:
            result = json.loads(llm_output)
            # print("result: ", result)
            category = result.get("category")
            direct_answer = result.get("direct_answer")
            retrieval_query = result.get("retrieval_query")
            # print("category: ", category, "direct_answer: ", direct_answer, "retrieval_query: ", retrieval_query)
        except Exception:
            # fallback if parsing fails
            category = "retrieval"
            direct_answer = None
            retrieval_query = user_query
        
        requires_retrieval = category == "retrieval"
        print("requires_retrieval: ", requires_retrieval)
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


def truncate_history(history, user_query, context_block, system_file="prompts/system_prompt.txt", history_file="prompts/history_prompt.txt", token_limit=TOKEN_LIMIT):
    try:
        with open(system_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        system_prompt = ""

    try:
        with open(history_file, "r", encoding="utf-8") as f:
            history_prompt = f.read()
    except FileNotFoundError:
        history_prompt = ""

    system_tokens = count_tokens(system_prompt)
    history_prompt_tokens = count_tokens(history_prompt)
    user_tokens = count_tokens(user_query)
    context_tokens = count_tokens(context_block)

    total_tokens = system_tokens + history_prompt_tokens + user_tokens + context_tokens

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
        # f.write("Prompt Sent to LLM:\n")
        # f.write(llm_prompt_text + "\n\n")
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