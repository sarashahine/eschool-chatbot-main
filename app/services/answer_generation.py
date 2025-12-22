import json

from app.services.utils import call_ollama_with_retries
from config import MODEL_NAME
from app.services.logging_utils import get_answer_generation_logger


def generate_answer(user_prompt, history_msgs, ollama_client, answer_generation_system_prompt, user_ip):
    messages = [{"role": "system", "content": answer_generation_system_prompt}]
    messages.extend(history_msgs)
    messages.append({"role": "user", "content": user_prompt})

    response = call_ollama_with_retries(
        lambda: ollama_client.chat(model=MODEL_NAME, messages=messages, format="json")
    )

    # Log full interaction for answer generation (with retrieved context + history)
    answer_logger = get_answer_generation_logger()
    try:
        prompt_payload = json.dumps(messages, ensure_ascii=False)
        raw_response = getattr(getattr(response, "message", None), "content", str(response))
        answer_logger.info(
            "",
            extra={
                "ip": user_ip,
                "prompt": prompt_payload,
                "response": raw_response,
            },
        )
    except Exception:
        # Logging must not break normal behavior
        pass

    try:
        content_json = json.loads(response.message.content)
        return content_json.get("answer", "")
    except Exception:
        return ""
