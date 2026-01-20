from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_rag_refund_window_e2e():
    r = client.post("/rag/ask", json = {"question": "How long is the refund window for Pro?", "top_k": 5})
    assert r.status_code == 200
    data = r.json()

    assert data["final_answer"]
    assert "14" in data["final_answer"]
    assert len(data["citations"]) >= 1

    cited_ids = {c["chunk_id"] for c in data["citations"]}
    assert "refund-policy::c0000" in cited_ids
    assert "trace_id" in data["retrieval_debug"]
    assert "timings_ms" in data["retrieval_debug"]