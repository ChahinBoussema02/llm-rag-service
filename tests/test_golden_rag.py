import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def _index_exists() -> bool:
    return Path("data/index/chroma").exists()

pytestmark = pytest.mark.skipif(
    not _index_exists(),
    reason = "Chroma index not found. Run: python -m app.rag.ingest && python -m app.rag.index"
)

def _load_goldens() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open("tests/golden_rag.jsonl", "r", encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items

@pytest.mark.parametrize("item", _load_goldens(), ids = lambda x: x.get("id", x.get("question", "golden")))
def test_golden_rag(item: Dict[str, Any]) -> None:
    payload = {"question": item["question"], "top_k": item.get("top_k", 5)}
    r = client.post("/rag/ask", json = payload)
    assert r.status_code == 200, r.text
    data = r.json()

    # Basic shape checks
    assert isinstance(data.get("final_answer"), str)
    assert isinstance(data.get("citations"), list)
    assert isinstance(data.get("retrieval_debug"), dict)

    answer = data["final_answer"].strip()
    citations = data["citations"]

    # Must not answer without citations (unless we explicitly expect IDK)
    if not item.get("must_say_idk", False):
        assert answer, "final_answer is empty"
        assert len(citations) >= 1, "expected at least 1 citation"

    # Must say IDK case
    if item.get("must_say_idk", False):
        assert "don't know" in answer.lower() , f"Expected IDK, got: {answer}"
        return
    
    cited_ids = {c.get("chunk_id") for c in citations}

    for cid in item.get("must_cite", []):
        assert cid in cited_ids, f"Missing required citation {cid}. Got: {cited_ids}"

    for s in item.get("must_contain", []):
        assert s.lower() in answer.lower(), f"Answer missing '{s}'. Got: {answer}"
