import os
import json
from typing import Any, Dict, List, AsyncIterator

import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", 180))

SYSTEM_PROMPT = """
You are a RAG assistant. Answer ONLY using the provided context chunks.
If the answer is not in the chunks, say: "I don't know based on the provided documents."
Be short and direct. Do NOT invent policies not in the context.
"""

def _build_user_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    ctx_lines = []
    for c in contexts:
        ctx_lines.append(f"[{c['chunk_id']}]\n{c['text']}\n")
    return f"""QUESTION:
    {question}

    CONTEXT CHUNKS:
    {''.join(ctx_lines)}
    """.strip()

async def stream_answer_text(question: str, contexts: List[Dict[str, Any]]) -> AsyncIterator[str]:
    """
    Streams plain text tokens from Ollama (/api/chat stream=true).
    This is for UX (typing effect). Your non-stream endpoint still returns strict JSON.
    """
    user_prompt = _build_user_prompt(question, contexts)

    payload = {
        "model": OLLAMA_MODEL,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": 0.1, "num_predict": 400},
    }

    async with httpx.AsyncClient(timeout = OLLAMA_TIMEOUT) as client:
        async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json = payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Ollama chat stream chunks often look like:
                # {"message":{"role":"assistant","content":"..."},"done":false}
                msg = (obj.get("message") or {})
                token = msg.get("content") or ""
                if token:
                    yield token

                if obj.get("done") is True:
                    break