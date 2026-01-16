# app/rag/generate.py
import json
import os
from typing import Any, Dict, List

import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))

SYSTEM_PROMPT = """
You are a RAG assistant. Answer ONLY using the provided context chunks.
If the answer is not in the chunks, say: "I don't know based on the provided documents."

You MUST output ONLY valid JSON with this shape:
{
  "final_answer": string,
  "used_chunk_ids": [string]
}

Rules:
- used_chunk_ids MUST be a subset of the chunk_ids provided.
- Prefer short, direct answers.
- Do NOT invent policies not in the context.
If you output anything other than JSON, your answer will be discarded.
"""


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Parse JSON even if the model adds extra text around it.
    """
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def generate_answer(question: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    contexts: list of {chunk_id, text, metadata, score}
    Returns: {"final_answer": str, "used_chunk_ids": [str], "warning": optional}
    """
    allowed_ids = [c["chunk_id"] for c in contexts]

    ctx_lines = []
    for c in contexts:
        ctx_lines.append(f"[{c['chunk_id']}]\n{c['text']}\n")

    user_prompt = f"""QUESTION:
{question}

CONTEXT CHUNKS:
{''.join(ctx_lines)}
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": 0.1, "num_predict": 300},
    }

    try:
        r = httpx.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        content = r.json()["message"]["content"]

        data = _extract_json(content)

        final_answer = str(data.get("final_answer", "")).strip()
        used = data.get("used_chunk_ids", [])
        if not isinstance(used, list):
            used = []

        # enforce subset
        used = [x for x in used if x in allowed_ids]

        # if model returned empty, treat as failure so we fall back
        if not final_answer:
            raise ValueError("Model returned empty final_answer")

        return {"final_answer": final_answer, "used_chunk_ids": used}

    except Exception as e:
        # Safe fallback so your API never crashes
        return {
            "final_answer": "I don't know based on the provided documents.",
            "used_chunk_ids": [],
            "warning": f"generation_failed: {e.__class__.__name__}",
        }