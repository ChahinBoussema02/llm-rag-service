from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import chromadb

from app.rag.embed import Embedder


def _tokens(s: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _keyword_boost(query: str, text: str) -> float:
    q = _tokens(query)
    t = _tokens(text)

    if not q or not t:
        return 0.0

    overlap = len(q & t) / max(1, len(q))

    # Extra boosts for common “policy answer” words
    bonus_terms = {"within", "eligible", "refund", "days", "window"}
    bonus = 0.05 * len((q | bonus_terms) & t)

    return overlap + bonus


@dataclass
class Retriever:
    persist_dir: Path
    collection_name: str = "company_docs"
    embedder: Optional[Embedder] = None

    def __post_init__(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        if self.embedder is None:
            self.embedder = Embedder()

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        assert self.embedder is not None
        q_emb = self.embedder.embed_query(query)

        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        out: List[Dict[str, Any]] = []
        for i in range(len(res["ids"][0])):
            chunk_id = res["ids"][0][i]
            text = res["documents"][0][i]
            meta = res["metadatas"][0][i]
            dist = float(res["distances"][0][i])

            # Chroma distance: smaller is better -> convert to similarity-ish
            score = 1.0 / (1.0 + dist)

            out.append(
                {
                    "chunk_id": chunk_id,
                    "score": score,
                    "text": text,
                    "metadata": meta,
                }
            )

        # Hybrid rerank: semantic score + keyword overlap/bonus
        for r in out:
            r["keyword_boost"] = _keyword_boost(query, r["text"])
            r["final_score"] = 0.75 * r["score"] + 0.25 * r["keyword_boost"]

            # Optional: section boost (good for policy Qs)
            sec = str(r["metadata"].get("section_path", "")).lower()
            if "eligibility" in sec:
                r["final_score"] += 0.08

        out.sort(key=lambda x: x["final_score"], reverse=True)
        return out
