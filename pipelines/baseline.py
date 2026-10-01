"""No-retrieval baseline: ask the model directly. Every later stack must beat this."""
import json
import os
import urllib.request

OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("LLM_MODEL", "qwen2.5-3b-safe")


def chat(messages: list[dict], model: str = MODEL) -> str:
    body = json.dumps({"model": model, "messages": messages, "stream": False,
                       "options": {"temperature": 0}}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["message"]["content"]


def run(question: str) -> dict:
    return {"answer": chat([{"role": "user", "content": question}]), "sources": []}
