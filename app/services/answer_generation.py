import json

from config import MODEL_NAME
from app.services.utils import call_ollama_with_retries
from app.services.logging_utils import log_answer_generation


def generate_answer(user_query, history_msgs, context_block, ollama_client, answer_generation_user_prompt, answer_generation_system_prompt, user_ip):
    messages = [{"role": "system", "content": answer_generation_system_prompt}]
    messages.extend(history_msgs)

    user_prompt = answer_generation_user_prompt.replace("{{CONTEXT_BLOCK}}", context_block).replace("{{USER_QUERY}}", user_query)
    messages.append({"role": "user", "content": user_prompt})

    response = call_ollama_with_retries(
        lambda: ollama_client.chat(model=MODEL_NAME, messages=messages, format="json")
    )

    # Flatten each chunk (split by double newlines) and keep chunks on separate lines
    chunks = context_block.split("\n\n")
    flattened_chunks = [" ".join(chunk.splitlines()).strip() for chunk in chunks]
    log_context = "\n".join(flattened_chunks)

    # single_line_context = " ".join(context_block.splitlines()).strip()
    try:
        raw_response = getattr(getattr(response, "message", None), "content", str(response))

        log_answer_generation(
                ip = user_ip,
                user_query = user_query,
                response = raw_response,
                prompt = messages,
                history = history_msgs,
        )
    except Exception:
        # Logging must not break normal behavior
        pass

    try:
        content_json = json.loads(response.message.content)
        return content_json.get("answer", "")
    except Exception:
        return ""
