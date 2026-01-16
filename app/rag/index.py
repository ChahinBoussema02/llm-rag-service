# app/rag/index.py
import json
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_chunks(chunks_path: Path) -> List[Dict]:
    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks

def build_chroma_index(
        chunks_path: Path,
        persist_dir: Path,
        collection_name: str = "company_docs",
        embedding_mode_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Tuple[int, str]:
    """
    Builds (or rebuilds) a Chroma collection with embeddings for each chunk.
    Returns: (num_chunks_indexed, persist_dir)
    """
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path = str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    #Recreate collection to keep builds deterministic
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    model = SentenceTransformer(embedding_mode_name)

    chunks = load_chunks(chunks_path)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict] = []

    for c in chunks:
        ids.append(c["chunk_id"])
        docs.append(c["text"])
        applies_to = c.get("applies_to", [])
        if isinstance(applies_to, list):
            applies_to_str = ", ".join(str(x) for x in applies_to)
        else:
            applies_to_str = str(applies_to)

        metas.append(
            {
                "doc_id": c["doc_id"],
                "title": c["title"],
                "category": c["category"],
                "version": c["version"],
                "last_updated": c["last_updated"],
                "applies_to": applies_to_str,
                "section_path": " > ".join(c["section_path"]),
                "source_file": c["source_file"],
                "start_line": int(c["start_line"]),
                "end_line": int(c["end_line"]),
            }
        )
    
    #Embeded in batches
    batch_size = 64
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_metas = metas[i:i+batch_size]

        embeddings = model.encode(batch_docs, normalize_embeddings=True).tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings,
        )

    return len(chunks), str(persist_dir)

def query_index(
    query: str,
    persist_dir: Path,
    collection_name: str = "company_docs",
    embedding_mode_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
)-> List[Dict]:
    """
    Returns top-k results: [{chunk_id, score, text, metadata}, ...]
    """
    client = chromadb.PersistentClient( path = str(persist_dir))

    collection = client.get_collection(name=collection_name)

    model = SentenceTransformer(embedding_mode_name)

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    response: List[Dict] = []
    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        document = results["documents"][0][i]
        metadata = results["metadatas"][0][i]
        distance =  results["distances"][0][i]

        score = float(1.0 / (1.0 + distance))

        response.append(
            {
                "chunk_id": chunk_id,
                "score": score,
                "text": document,
                "metadata": metadata,
            }
        )
    
    return response

if __name__ == "__main__":
    root = _repo_root()
    chunks_path = root / "data" / "processed" / "chunks.jsonl"
    persist_dir = root / "data" / "index" / "chroma"

    n, p = build_chroma_index(chunks_path, persist_dir)
    print(f"Indexed {n} chunks into Chroma at: {p}")