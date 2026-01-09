from flask import Blueprint, request, jsonify, render_template

from flask import current_app
from config import HISTORY_START_MESSAGE, HISTORY_END_MESSAGE, BLOCK_THRESHOLD, BLOCK_MESSAGE
from ..arabic_services.utils import OllamaUnavailable
from ..arabic_services.arabic_retrieval import retrieve, truncate_answer_generation_history, pre_process_query
from ..arabic_services.arabic_answer_generation import generate_answer

arabic_main_routes = Blueprint("arabic_main", __name__)

# -----------------------------
# HTML route
# -----------------------------
@arabic_main_routes.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Query route
# -----------------------------
@arabic_main_routes.route("/query", methods=["POST"])
def query():
    try:
        data = request.json
        user_query = data.get("query", "")
        history = data.get("history", [])
        unrelated_streak = int(data.get("unrelated_streak", 0))
        user_ip = request.remote_addr or "unknown"

        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        decision = pre_process_query(
            user_query,
            current_app.decision_making_user_prompt,
            history,
            current_app.ollama_client,
            current_app.decision_making_system_prompt,
            user_ip=user_ip,
        )

        requires_retrieval = (decision["category"] == "related")
        answer = decision["answer"]
        category = decision["category"]
        unrelated_streak = unrelated_streak + 1 if category == "unrelated" else 0

        if unrelated_streak >= BLOCK_THRESHOLD:
            return jsonify({"answer": BLOCK_MESSAGE, "blocked": True})

        if not requires_retrieval:
            history.append({"question": user_query, "answer": answer})
            return jsonify({"answer": answer, "blocked": False})

        results = retrieve(user_query, current_app.embedder, current_app.qdrant_client)
        context_block = "\n\n".join(
            f" Text: {r['text']}\n Section: {r['metadata'].get('section_title','')}\n URL: {r['metadata'].get('url','')} . "
            for r in results
        )

        history = truncate_answer_generation_history(
            current_app.answer_generation_system_prompt_tokens,
            user_query,
            context_block,
            history,
            current_app.ollama_client
        )

        history_block = ""
        if history:
            history_block = HISTORY_START_MESSAGE
            for h in history:
                history_block += (
                    f"  User: {h['question']}\n"
                    f"  Assistant: {h['answer']}\n"
                )
            history_block += HISTORY_END_MESSAGE

        answer = generate_answer(
            user_query,
            history_block,
            context_block,
            current_app.ollama_client,
            current_app.answer_generation_user_prompt,
            current_app.answer_generation_system_prompt,
            user_ip=user_ip
        )

        history.append({"question": user_query, "answer": answer})

        return jsonify({"answer": answer, "blocked": False, "unrelated_streak": unrelated_streak})

    except OllamaUnavailable:
        # Specific, user-friendly message when Ollama keeps failing
        return jsonify({
            "error": "The AI model is temporarily unavailable. Please try again in a little while."
        }), 503
    except Exception as e:
        # Fallback for any other unexpected server error
        return jsonify({"error": "Unexpected server error."}), 500
