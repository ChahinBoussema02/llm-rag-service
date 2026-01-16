from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from sentence_transformers import SentenceTransformer


@dataclass
class Embedder:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    _model: Optional[SentenceTransformer] = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model 

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        embs = model.encode(texts, normalize_embeddings=True)
        return embs.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]