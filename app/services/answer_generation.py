import json

from config import MODEL_NAME, RETRIEVAL_START_MESSAGE, RETRIEVAL_END_MESSAGE
from app.services.utils import call_ollama_with_retries
from app.services.logging_utils import log_answer_generation


def generate_answer(user_query, history, context_block, ollama_client, answer_generation_user_prompt, answer_generation_system_prompt, user_ip):
    messages = [{"role": "system", "content": answer_generation_system_prompt}]

    # Build a single user payload with clear hierarchy
    user_content_parts = []

    if history:
        user_content_parts.append(history)

    user_content_parts.append(RETRIEVAL_START_MESSAGE)
    user_content_parts.append(context_block)
    user_content_parts.append(RETRIEVAL_END_MESSAGE)

    user_prompt = answer_generation_user_prompt.replace("{{USER_QUERY}}", user_query)
    user_content_parts.append(user_prompt)

    messages.append({"role": "user", "content": "\n\n".join(user_content_parts)})

    response = call_ollama_with_retries(
        lambda: ollama_client.chat(model=MODEL_NAME, messages=messages, format="json")
    )

    # Flatten each chunk (split by double newlines) and keep chunks on separate lines
    # chunks = context_block.split("\n\n")
    # flattened_chunks = [" ".join(chunk.splitlines()).strip() for chunk in chunks]
    # log_context = "\n".join(flattened_chunks)

    # single_line_context = " ".join(context_block.splitlines()).strip()
    try:
        raw_response = getattr(getattr(response, "message", None), "content", str(response))

        log_answer_generation(
                ip = user_ip,
                user_query = user_query,
                response = raw_response,
        )
    except Exception:
        # Logging must not break normal behavior
        pass

    try:
        content_json = json.loads(response.message.content)
        return content_json.get("answer", "")
    except Exception:
        return ""
