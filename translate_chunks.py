import json
from app import create_app
from flask import current_app

from config import MODEL_NAME, TEXT_TO_EMBED_PATH, ARABIC_TEXT_TO_EMBED_PATH


def translate_using_deepseek_ollama(prompt, input_file=TEXT_TO_EMBED_PATH, output_file=ARABIC_TEXT_TO_EMBED_PATH):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []

    for i, item in enumerate(data):
        print("item: ",i)
        messages = [{"role": "user", "content": prompt + item["text"]}]
        response = current_app.ollama_client.chat(model=MODEL_NAME, messages=messages)
        item["text"] = getattr(getattr(response, "message", None), "content", str(response))
        results.append(item)
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processing complete. Results saved to {output_file}")


app = create_app()

with app.app_context():
    prompt = current_app.translation_prompt

    translate_using_deepseek_ollama(prompt)
