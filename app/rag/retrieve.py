from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from rank_bm25 import BM25Okapi # pyright: ignore[reportMissingImports]

import chromadb

from app.rag.embed import Embedder

def _tokens(s: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))

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

    def _tokenize(self, s: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", s.lower())
    
    
    def _load_all_docs(self) -> List[Dict[str, Any]]:
        """
        Pull all docs from Chroma for BM25.
        Cached after first call.
        """
        if hasattr(self, "_bm25_cache"):
            return self._bm25_cache
        
        # Get everything (Chroma supports fetching all stored docs)
        data = self.collection.get(include = ["documents", "metadatas"])
        docs: List[Dict[str, Any]] = []
        for i, chunk_id in enumerate(data["ids"]):
            docs.append({
                "chunk_id": chunk_id,
                "text": data["documents"][i],
                "metadata": data["metadatas"][i] or {}

            })
        self._bm25_cache = docs # type: ignore[attr-defined]
        return docs
    
    def _keyword_boost(self, query: str, text: str) -> float:
        q = _tokens(query)
        t = _tokens(text)

        if not q or not t:
            return 0.0

        overlap = len(q & t) / max(1, len(q))

        # Extra boosts for common “policy answer” words
        bonus_terms = {"within", "eligible", "refund", "days", "window"}
        bonus = 0.05 * len((q | bonus_terms) & t)

        if "refund" in query.lower() and "refund" in text.lower():
            return overlap + bonus + 0.2
        
        return overlap + bonus
        
    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        docs = self._load_all_docs()

        tokenized_corpus = [self._tokenize(d["text"]) for d in docs]
        bm25 = BM25Okapi(tokenized_corpus)

        scores = bm25.get_scores(self._tokenize(query))
        # top indices
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        out = []
        for i in ranked:
            out.append(
                {
                    "chunk_id": docs[i]["chunk_id"],
                    "text": docs[i]["text"],
                    "metadata": docs[i]["metadata"],
                    "bm25_score": float(scores[i]),
                }
            )
        return out

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        assert self.embedder is not None

        # 1) Vector search (grab more than top_k so fusion has room)
        vec_k = max(10, top_k * 3)
        q_emb = self.embedder.embed_query(query)

        vec = self.collection.query(
            query_embeddings=[q_emb],
            n_results=vec_k,
            include=["documents", "metadatas", "distances"],
        )

        vec_out: List[Dict[str, Any]] = []
        for i in range(len(vec["ids"][0])):
            chunk_id = vec["ids"][0][i]
            text = vec["documents"][0][i]
            meta = vec["metadatas"][0][i] or {}
            dist = float(vec["distances"][0][i])

            score = 1.0 / (1.0 + dist)  # normalize distance → similarity-ish
            vec_out.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": meta,
                    "vec_score": score,
                }
            )

        # 2) BM25 search
        bm25_k = max(10, top_k * 3)
        bm25_out = self._bm25_search(query, top_k=bm25_k)

        # Normalize BM25 scores to [0,1] (simple max-normalization)
        bm_max = max([r["bm25_score"] for r in bm25_out], default=1.0) or 1.0
        for r in bm25_out:
            r["bm25_norm"] = r["bm25_score"] / bm_max

        # 3) Merge by chunk_id
        merged: Dict[str, Dict[str, Any]] = {}

        def upsert(r: Dict[str, Any]) -> None:
            cid = r["chunk_id"]
            if cid not in merged:
                merged[cid] = {
                    "chunk_id": cid,
                    "text": r.get("text", ""),
                    "metadata": r.get("metadata", {}) or {},
                    "vec_score": 0.0,
                    "bm25_norm": 0.0,
                }
            merged[cid]["text"] = merged[cid]["text"] or r.get("text", "")
            if r.get("metadata"):
                merged[cid]["metadata"] = r["metadata"]
            if "vec_score" in r:
                merged[cid]["vec_score"] = max(merged[cid]["vec_score"], float(r["vec_score"]))
            if "bm25_norm" in r:
                merged[cid]["bm25_norm"] = max(merged[cid]["bm25_norm"], float(r["bm25_norm"]))

        for r in vec_out:
            upsert(r)
        for r in bm25_out:
            upsert(r)

        out = list(merged.values())

        # 4) Optional keyword boost (your idea)
        for r in out:
            kb = self._keyword_boost(query, r["text"]) # type: ignore
            r["keyword_boost"] = kb

        # 5) Final fusion score
        # weights: vectors 0.60, bm25 0.30, keyword boost 0.10
        for r in out:
            r["final_score"] = (
                0.55 * r["vec_score"]
                + 0.30 * r["bm25_norm"]
                + 0.15 * r["keyword_boost"]
            )
            r["score"] = r["final_score"]

        
        out.sort(key=lambda x: x["final_score"], reverse=True)
        return out[:top_k]
