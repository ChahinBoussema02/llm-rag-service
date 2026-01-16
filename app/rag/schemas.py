# app/rag/schemas.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    section_path: str
    score: float
    snippet: str

class AskRagRequest(BaseModel):
    question: str = Field(min_length = 1, max_length = 2000)
    top_k: int = Field(default = 5, ge = 1, le = 10)

class AskRagResponse(BaseModel):
    question: str
    final_answer: str
    citations: List[Citation]
    retrieval_debug : Dict[str, Any]
