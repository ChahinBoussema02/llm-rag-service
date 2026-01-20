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

    # If generation failed or produced empty answer, treat as IDK
    if not final_answer:
        final_answer = "I don't know based on the provided documents."

    used = set(gen.get("used_chunk_ids", []))

    if _is_idk(final_answer):
        citations = []
    else:
        selected = [r for r in results if r["chunk_id"] in used]

        # fallback if model didn't pick any
        if not selected:
            selected = [results[0]]

        # optionally cap to 2
        selected = selected[:2]

        citations = [
            Citation(
                chunk_id=r["chunk_id"],
                doc_id=r["metadata"]["doc_id"],
                section_path=r["metadata"]["section_path"],
                score=r["score"],
                snippet=r["text"][:220],
            )
            for r in selected
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