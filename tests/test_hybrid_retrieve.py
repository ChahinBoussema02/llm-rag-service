from pathlib import Path
from app.rag.retrieve import Retriever

def test_hybrid_retrieval_mentions_refund():
    r = Retriever(Path("data/index/chroma"))
    out = r.search("refund window pro", top_k=3)

    assert len(out) >= 1
    joined = " ".join([x["text"].lower() for x in out])
    assert "refund" in joined