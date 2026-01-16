from pathlib import Path

import pytest

from app.rag.retrieve import Retriever

@pytest.mark.integration
def test_retriever_reranks_refund_eligibility_top():
    """
    For refund-window questions, we expect the Eligibility chunk to outrank
    'How to Request' chunks after hybrid reranking.
    """
    index_dir = Path("data/index/chroma")
    if not index_dir.exists():
        pytest.skip("Chroma index not found. Run: python -m app.rag.index")

    r = Retriever(index_dir)
    results = r.search("refund window pro", top_k = 5)

    # sanity
    assert len(results) >= 2

    top_ids = [x["chunk_id"] for x in results[:3]]

    # We want the Eligibility chunk to be very near the top
    assert "refund-policy::c0000" in top_ids, f"Top-3 were: {top_ids}"