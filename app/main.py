from dotenv import load_dotenv
load_dotenv()

import time
import uuid
import logging

from pathlib import Path
from fastapi import FastAPI
from typing import Optional, List, Dict, Any

from app.rag.schemas import AskRagRequest, AskRagResponse, Citation
from app.rag.generate import generate_answer
from app.rag.retrieve import Retriever

from cachetools import TTLCache

retrieval_cache = TTLCache(maxsize=512, ttl=300)  # 5 minutes

app = FastAPI(title="LLM RAG Service")

logger = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO)

CHROMA_DIR = Path("data/index/chroma")

# Create retriever once (keeps app fast)
retriever = Retriever(CHROMA_DIR)


@app.get("/health")
def health():
    return {"ok": True}


def _is_idk(text: str) -> bool:
    t = (text or "").strip().lower()
    return "don't know" in t or "do not know" in t

def infer_category(question: str) -> Optional[str]:
    q = question.lower()
    if any(k in q for k in ["refund", "billing", "plan", "pricing", "downgrade", "upgrade", "past due"]):
        return "billing"
    if any(k in q for k in ["privacy", "retention", "delete", "gdpr", "data"]):
        return "privacy"
    if any(k in q for k in ["support", "response time", "sla", "ticket"]):
        return "support"
    if any(k in q for k in ["incident", "outage", "status", "downtime"]):
        return "operations"
    return None

def filter_results(
        results: List[Dict[str, Any]],
        category: Optional[str],
        applies_to: Optional[str],
)-> List[Dict[str, Any]]:
    out = []
    for r in results:
        meta = r.get("metadata", {}) or {}

        if category and meta.get("category") != category:
            continue
        
        if applies_to:
            # your metadata stores applies_to like "Pro, Team" (string)
            allowed = str(meta.get("applies_to", ""))
            if applies_to.lower() not in allowed.lower():
                continue
        
        out.append(r)
    return out

def _pick_best_chunk_for_question(question: str, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    q = question.lower()

    # Heuristic: refund window questions → prefer chunks mentioning "within" + "days"
    if "refund" in q and ("window" in q or "how long" in q or "eligible" in q):
        for r in results:
            t = r["text"].lower()
            if "eligible" in t and "within" in t and "days" in t:
                return r

    # Heuristic: "request a refund" → prefer chunk containing "Refund Request"
    if "request a refund" in q or ("how do i" in q and "refund" in q):
        for r in results:
            if "refund request" in r["text"].lower():
                return r

    # default: best-scoring
    return results[0] if results else None


def _extract_answer_from_chunk(chunk_text: str, max_chars: int = 220) -> str:
    """
    Simple extractive answer:
    - take first 1–2 bullet lines or first sentence
    """
    lines = [ln.strip() for ln in chunk_text.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Prefer bullet lines
    bullets = [ln for ln in lines if ln.startswith("-")]
    if bullets:
        # Join first 1–2 bullets
        out = bullets[0]
        if len(bullets) > 1 and len(out) < 120:
            out += " " + bullets[1]
        return out[:max_chars].strip()

    # Otherwise first line
    return lines[0][:max_chars].strip()

@app.post("/rag/ask", response_model=AskRagResponse)
def ask_rag(req: AskRagRequest):
    t0 = time.perf_counter()
    trace_id = str(uuid.uuid4())
    results = retriever.search(req.question, top_k=req.top_k)

    effective_category = req.category or infer_category(req.question)
    results = filter_results(results, effective_category, req.applies_to)

    top_score = results[0]["score"] if results else 0.0

    # Evidence sufficiency gate (prevents hallucinations)
    if not results or top_score < 0.45:
        return AskRagResponse(
            question=req.question,
            final_answer="I don't know based on the provided documents.",
            citations=[],
            retrieval_debug={
                "top_k": req.top_k,
                "results": results,
                "reason": "low_retrieval_confidence",
                "top_score": top_score,
            },
        )
    t_retrieve = time.perf_counter()
    retrieve_ms = int((t_retrieve - t0) * 1000)

    gen = generate_answer(req.question, results) or {}
    final_answer = (gen.get("final_answer") or "").strip()
    gen_warning = gen.get("warning")

    # If Ollama is missing/down in CI (or any generation failure), do extractive fallback
    if gen_warning:
        best = _pick_best_chunk_for_question(req.question, results)
        if best:
            final_answer = _extract_answer_from_chunk(best["text"])
            # force citations to match the fallback evidence
            used = {best["chunk_id"]}
        else:
            final_answer = "I don't know based on the provided documents."
            used = set()
    else:
        used = set(gen.get("used_chunk_ids", []))

    t_gen = time.perf_counter()
    gen_ms = int((t_gen - t_retrieve) * 1000)
    total_ms = int((t_gen - t0) * 1000)

    cache_key = (req.question, req.top_k, effective_category, req.applies_to)

    cached = retrieval_cache.get(cache_key)
    if cached is not None:
        results = cached
        cache_hit = True
    else:
        results = retriever.search(req.question, top_k=req.top_k)
        results = filter_results(results, effective_category, req.applies_to)
        retrieval_cache[cache_key] = results
        cache_hit = False


    used = set(gen.get("used_chunk_ids", []))

    if _is_idk(final_answer):
        citations = []
    else:
        # If model provided used_chunk_ids OR fallback picked one, cite those.
        if used:
            selected = [r for r in results if r["chunk_id"] in used]
            # safety: if somehow none matched, cite top1
            if not selected:
                selected = [results[0]]
        else:
            selected = [results[0]]

        citations = [
            Citation(
                chunk_id=r["chunk_id"],
                doc_id=r["metadata"]["doc_id"],
                section_path=r["metadata"]["section_path"],
                score=r["score"],
                snippet=r["text"][:220],
            )
            for r in selected[:2]
        ]

    logger.info(
    "rag_request trace_id=%s question_len=%d category=%s applies_to=%s top_k=%d top_score=%.3f idk=%s retrieve_ms=%d gen_ms=%d total_ms=%d cited=%d",
    trace_id, len(req.question), effective_category, req.applies_to, req.top_k,
    top_score, _is_idk(final_answer), retrieve_ms, gen_ms, total_ms, len(citations),
)

    return AskRagResponse(
        question=req.question,
        final_answer=final_answer,
        citations=citations,
        retrieval_debug={
            "top_k": req.top_k,
            "results": results,
            "top_score": top_score,
            "gen_warning": gen.get("warning"),
            "category": effective_category,
            "applies_to": req.applies_to,
            "trace_id": trace_id,
            "timings_ms": {"retrieve": retrieve_ms, "generate": gen_ms, "total": total_ms},
        },
    )