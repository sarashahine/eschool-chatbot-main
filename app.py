from flask import Flask, request, jsonify, render_template
import json
import os
import re
from dotenv import load_dotenv
from ollama import Client
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.http import models as rest

from config import COLLECTION_NAME, QDRANT_HTTP, EMBEDDING_MODEL, TOP_K, MODEL_NAME, TOKEN_LIMIT

load_dotenv()


# -----------------------------
# Initialize
# -----------------------------
app = Flask(__name__)
model = SentenceTransformer(EMBEDDING_MODEL)
OLLAMA_KEY = os.getenv("OLLAMA_API_KEY")
if not OLLAMA_KEY:
    raise ValueError("OLLAMA_API_KEY is missing in .env")
ollama_client = Client(
                    host="https://ollama.com",
                    headers={'Authorization': 'Bearer ' + OLLAMA_KEY}
                )
qdrant_client = QdrantClient(url=QDRANT_HTTP)


# -----------------------------
# Global auto-increment ID
# -----------------------------
NEXT_CHUNK_ID = None

def init_next_chunk_id():
    global NEXT_CHUNK_ID
    try:
        max_id = 0
        offset = None
        while True:
            points, offset = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                with_payload=False,
                with_vectors=False,
                offset=offset,
            )
            if not points:
                break
            for p in points:
                try:
                    max_id = max(max_id, int(p.id))
                except Exception:
                    # Ignore non-numeric ids
                    continue
            if offset is None:
                break
        NEXT_CHUNK_ID = max_id + 1
    except Exception:
        # safest fallback
        NEXT_CHUNK_ID = 1

init_next_chunk_id()


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



def safe_json_load(llm_output: str, user_query):
    cleaned = re.sub(r"^```json\s*|```$", "", llm_output.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
        if "category" not in data or "answer" not in data:
            raise ValueError("Missing expected keys")
        return data
    except Exception:
        print("Failed to parse JSON. Cleaned output:", repr(cleaned))
        return {
            "category": "retrieval",
            "answer": ""
        }
    
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

        result = safe_json_load(llm_output, user_query)
        
        print("result: ", result)
        category = result.get("category")
        answer = result.get("answer", "")

        requires_retrieval = (category == "retrieval")
        print("requires_retrieval: ", requires_retrieval)

        retrieval_query = user_query if requires_retrieval else None

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
            "direct_answer": answer if not requires_retrieval else None,
            "retrieval_query": retrieval_query
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
        return result.get("total_tokens", len(text.split()))  # fallback if missing
    except Exception:
        # fallback in case API fails
        return len(text.split())


def truncate_history(history, user_query, context_block, token_limit=TOKEN_LIMIT):

    system_tokens = count_tokens(SYSTEM_PROMPT_BASE)
    user_tokens = count_tokens(user_query)
    context_tokens = count_tokens(context_block)

    total_tokens = system_tokens + user_tokens + context_tokens

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
    separator = "\n\n\n" + "="*100 + "\n\n\n\n"
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
    # print(f"Interaction logged to '{log_file}'")


def generate_with_deepseek(system_prompt: str, user_prompt: str, history_turns: list):
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    for turn in history_turns:
        messages.append(turn)
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = ollama_client.chat(model=MODEL_NAME, messages=messages)
        answer = response.message.content
    except Exception as e:
        print("Error during LLM call:", e)
        raise e
    
    print("answer generated")

    full_prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    log_llm_interaction(llm_prompt_text=full_prompt_text, llm_answer=answer, history=history_turns)
    
    return answer

    
# -----------------------------
# Flask routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.json
        user_query = data.get("query", "")
        history = data.get("history", [])

        if not user_query:
            return jsonify({"error": "No query provided"}), 400


        decision = pre_process_query(user_query, history)
        requires_retrieval = decision["requires_retrieval"]
        direct_answer = decision["direct_answer"]
        retrieval_query = decision["retrieval_query"]

        if not requires_retrieval:
            answer = direct_answer

            history.append({
                "question": user_query,
                "answer": answer
            })

            return jsonify({
                "answer": answer,
                "history": history ### delete this later
            })


        print("retrieval_query:", retrieval_query)
        results = retrieve(retrieval_query, top_k=TOP_K)

        context_block = "\n\n".join(
            f"Text: {r['text']}\nSection: {r['metadata'].get('section_title','')}\nURL: {r['metadata'].get('url','')}"
            for r in results
        )

        history = truncate_history(history, user_query, context_block)

        history_entries = []
        for turn in history:
            if turn.get("question"):
                history_entries.append({"role": "user", "content": turn["question"]})
            if turn.get("answer"):
                history_entries.append({"role": "assistant", "content": turn["answer"]})

        system_prompt = SYSTEM_PROMPT_BASE
        
        user_prompt = f"""
    Context:
    {context_block}

    Question:
    {user_query}
    Question Reformulated:
    {retrieval_query}

    Answer:
    """
        
        print("Preparing to call generate_with_deepseek...")
        answer = generate_with_deepseek(system_prompt, user_prompt, history_entries)
        print("LLM call succeeded, answer length:", len(answer))
        history.append({
            "question": user_query,
            "answer": answer
        })

        return jsonify({
            "answer": answer,
            "history": history
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chunks", methods=["POST"])
def insert_chunk():
    try:
        data = request.json or {}
        text = (data.get("text") or "").strip()
        metadata = data.get("metadata") or {}

        if not text:
            return jsonify({"error": "'text' is required."}), 400

        # Ensure NEXT_CHUNK_ID is initialized
        global NEXT_CHUNK_ID
        if NEXT_CHUNK_ID is None:
            init_next_chunk_id()

        chunk_id = NEXT_CHUNK_ID
        NEXT_CHUNK_ID += 1

        # Validate metadata
        allowed_keys = {"page_title", "url", "section_title"}
        filtered_metadata = {k: v for k, v in metadata.items() if k in allowed_keys}

        vector = model.encode(text, convert_to_tensor=False).tolist()
        payload = {"text": text, **filtered_metadata}
        point = rest.PointStruct(id=int(chunk_id), vector=vector, payload=payload)
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[point], wait=True)

        return jsonify({"message": f"Inserted chunk {chunk_id}.", "id": chunk_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chunks/<int:chunk_id>", methods=["GET"])
def get_chunk(chunk_id: int):
    try:
        existing = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[chunk_id],
            with_vectors=False,
        )
        if not existing:
            return jsonify({"error": f"Chunk {chunk_id} not found."}), 404
        
        record = existing[0]
        payload = dict(record.payload or {})
        return jsonify({
            "id": chunk_id,
            "text": payload.get("text", ""),
            "metadata": {
                "page_title": payload.get("page_title", ""),
                "url": payload.get("url", ""),
                "section_title": payload.get("section_title", ""),
            },
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/chunks/<int:chunk_id>", methods=["PUT"])
def update_chunk(chunk_id: int):
    try:
        data = request.json or {}
        new_text = data.get("text")
        metadata_updates = data.get("metadata")

        existing = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[chunk_id],
            with_vectors=True
        )

        if not existing:
            return jsonify({"error": f"Chunk {chunk_id} not found."}), 404

        record = existing[0]
        current_payload = dict(record.payload or {})
        vector = record.vector

        if new_text is not None:
            new_text = new_text.strip()
            vector = model.encode([new_text], convert_to_tensor=False)[0].tolist()
            current_payload["text"] = new_text

        if metadata_updates:
            allowed_keys = {"page_title", "url", "section_title"}
            filtered_updates = {k: v for k, v in metadata_updates.items() if k in allowed_keys}
            current_payload.update(filtered_updates)

        updated_point = rest.PointStruct(
            id=chunk_id,
            vector=vector,
            payload=current_payload
        )
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=[updated_point], wait=True)

        return jsonify({"message": f"Updated chunk {chunk_id}."})
    
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/chunks/<int:chunk_id>", methods=["DELETE"])
def delete_chunk(chunk_id: int):
    try:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=rest.PointIdsList(points=[chunk_id]),
            wait=True
        )
        return jsonify({"message": f"Deleted chunk {chunk_id}."})
    
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)