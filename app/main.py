from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from fastapi import FastAPI

from app.rag.schemas import AskRagRequest, AskRagResponse, Citation
from app.rag.generate import generate_answer
from app.rag.retrieve import Retriever

app = FastAPI(title="LLM RAG Service")

CHROMA_DIR = Path("data/index/chroma")

# Create retriever once (keeps app fast)
retriever = Retriever(CHROMA_DIR)


@app.get("/health")
def health():
    return {"ok": True}


def _is_idk(text: str) -> bool:
    t = (text or "").strip().lower()
    return "don't know" in t or "do not know" in t


@app.post("/rag/ask", response_model=AskRagResponse)
def ask_rag(req: AskRagRequest):
    results = retriever.search(req.question, top_k=req.top_k)

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

    gen = generate_answer(req.question, results) or {}
    final_answer = (gen.get("final_answer") or "").strip()

    # If generation failed or produced empty answer, treat as IDK
    if not final_answer:
        final_answer = "I don't know based on the provided documents."

    # If IDK → no citations (prevents noisy/false grounding)
    if _is_idk(final_answer):
        citations = []
    else:
        # Citation selection: top-1 always; top-2 only if strongly related
        top1 = results[0]
        selected = [top1]

        if len(results) > 1:
            top2 = results[1]
            same_doc = top2["metadata"].get("doc_id") == top1["metadata"].get("doc_id")
            same_category = top2["metadata"].get("category") == top1["metadata"].get("category")
            if same_doc or same_category:
                selected.append(top2)

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

    return AskRagResponse(
        question=req.question,
        final_answer=final_answer,
        citations=citations,
        retrieval_debug={
            "top_k": req.top_k,
            "results": results,
            "top_score": top_score,
            "gen_warning": gen.get("warning"),
        },
    )