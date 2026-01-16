import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi.testclient import TestClient

from app.main import app


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="eval/dataset.jsonl")
    parser.add_argument("--out", type=str, default="eval/results.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)

    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    # Fast check for index existence
    index_dir = Path("data/index/chroma")
    if not index_dir.exists():
        raise SystemExit("Chroma index not found. Run: python -m app.rag.ingest && python -m app.rag.index")

    items = load_jsonl(dataset_path)
    client = TestClient(app)

    results: List[Dict[str, Any]] = []

    # Metrics counters
    n = 0
    answered = 0
    said_idk = 0
    has_citations = 0
    must_cite_total = 0
    must_cite_hits = 0
    contain_total = 0
    contain_hits = 0

    for item in items:
        n += 1
        q = item["question"]
        top_k = int(item.get("top_k", 5))

        r = client.post("/rag/ask", json={"question": q, "top_k": top_k})
        ok = r.status_code == 200

        if not ok:
            results.append(
                {
                    "id": item.get("id"),
                    "question": q,
                    "ok": False,
                    "status_code": r.status_code,
                    "error": r.text,
                }
            )
            continue

        data = r.json()
        answer = (data.get("final_answer") or "").strip()
        citations = data.get("citations") or []
        cited_ids = {c.get("chunk_id") for c in citations if isinstance(c, dict)}

        if answer:
            answered += 1
        if "don't know" in answer.lower():
            said_idk += 1
        if len(citations) > 0:
            has_citations += 1

        # Checks (optional fields in dataset)
        example_checks: Dict[str, Any] = {}

        must_cite = item.get("must_cite", [])
        if must_cite:
            must_cite_total += 1
            hit = all(cid in cited_ids for cid in must_cite)
            must_cite_hits += int(hit)
            example_checks["must_cite"] = {"required": must_cite, "hit": hit, "got": sorted(list(cited_ids))}

        must_contain = item.get("must_contain", [])
        if must_contain:
            contain_total += 1
            hit = all(s.lower() in answer.lower() for s in must_contain)
            contain_hits += int(hit)
            example_checks["must_contain"] = {"required": must_contain, "hit": hit}

        must_say_idk = bool(item.get("must_say_idk", False))
        if must_say_idk:
            example_checks["must_say_idk"] = {"required": True, "hit": ("don't know" in answer.lower())}

        results.append(
            {
                "id": item.get("id"),
                "question": q,
                "ok": True,
                "answer": answer,
                "citations": citations,
                "checks": example_checks,
            }
        )

    # Compute metrics
    metrics = {
        "n": n,
        "answer_rate": answered / n if n else 0.0,
        "idk_rate": said_idk / n if n else 0.0,
        "citation_rate": has_citations / n if n else 0.0,
        "must_cite_hit_rate": (must_cite_hits / must_cite_total) if must_cite_total else None,
        "contain_hit_rate": (contain_hits / contain_total) if contain_total else None,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"metrics": metrics, "results": results}, indent=2), encoding="utf-8")

    print("Eval complete ✅")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()