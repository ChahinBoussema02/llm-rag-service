# LLM RAG Service (Local Ollama + Chroma) ✅

A **portfolio-ready FastAPI RAG service** that demonstrates a **production-style Retrieval-Augmented Generation (RAG)** pipeline using:

- **Local LLM via Ollama** (default: `gemma3:12b`)
- **ChromaDB** for vector search
- **SentenceTransformers** embeddings
- **Hybrid retrieval + reranking**
- **Strict citations + evaluation** to reduce hallucinations

---

## ✨ What this service does

You send a question → it retrieves the most relevant document chunks → the LLM answers **only using those chunks** → it returns:

- a short final answer
- citations (chunk ids + snippets)
- retrieval debug output (for inspection and evaluation)

Example response:

```json
{
  "question": "How long do Pro users have to request a refund?",
  "final_answer": "Pro and Team plans are eligible for refunds within 14 days of the initial purchase.",
  "citations": [
    {
      "chunk_id": "refund-policy::c0000",
      "doc_id": "refund-policy",
      "section_path": "Refund Policy (AcmeAI) > Eligibility",
      "score": 0.64,
      "snippet": "- Pro and Team plans are eligible for refunds within 14 days..."
    }
  ],
  "retrieval_debug": {
    "top_k": 5,
    "results": []
  }
}
```

---

## 🧠 Architecture (High-level)

**Pipeline:**

1. **Ingest docs** (`.md` with metadata frontmatter)
2. **Chunk docs** into `chunks.jsonl`
3. **Embed + index** into **ChromaDB**
4. **Retrieve** top-k chunks (**hybrid search + reranking**)
5. **Generate answer** with **Gemma 3 12B** (Ollama)
6. Return **final answer + citations**

---

## 🔧 Tech Stack

- **FastAPI** (HTTP API)
- **Ollama** (local model runtime)
- **Gemma 3 12B** (`gemma3:12b`)
- **ChromaDB** (vector DB)
- **SentenceTransformers** (embeddings)
- **Rank-BM25** (keyword retrieval)
- **Pytest** (tests)
- **Eval harness** (`eval/run_eval.py`)
- **Docker + docker-compose**
- **GitHub Actions CI**

---

## 📦 Project Structure

```bash
llm-rag-service/
  app/
    main.py                  # FastAPI app
    rag/
      ingest.py              # parse + chunk docs
      index.py               # build Chroma index
      retrieve.py            # Retriever (hybrid retrieval + reranking)
      embed.py               # SentenceTransformer embedder
      generate.py            # Ollama generation (strict JSON output)
      schemas.py             # Pydantic request/response models
  data/
    raw/company_docs/        # input markdown docs (source KB)
    processed/chunks.jsonl   # chunked docs (output of ingest)
    index/chroma/            # chroma persistent index (output of index)
  tests/
    test_golden_rag.py       # golden API tests
    test_retrieve.py         # retrieval quality test
    test_routing.py          # routing + filtering tests
  eval/
    dataset.jsonl            # evaluation dataset
    run_eval.py              # runs eval + metrics
  .github/workflows/
    ci.yml                   # GitHub Actions CI
  Dockerfile
  docker-compose.yml
  Makefile
  requirements.txt
  README.md
```

---

## 🚀 Quickstart (Local)

### 1) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🦙 Run Ollama (Gemma 3 12B)

Install Ollama from: https://ollama.com

Pull the model:

```bash
ollama pull gemma3:12b
```

Verify it is available:

```bash
ollama list
```

---

## ⚙️ Environment Variables

Create a `.env` file in the project root:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:12b
OLLAMA_TIMEOUT_SECONDS=180
```

---

## 📚 Step 1 — Ingest docs into chunks

This reads markdown docs from:

```bash
data/raw/company_docs/
```

and writes:

```bash
data/processed/chunks.jsonl
```

Run:

```bash
python -m app.rag.ingest
```

---

## 🧱 Step 2 — Build the Chroma index

This reads:

```bash
data/processed/chunks.jsonl
```

and writes the persistent index to:

```bash
data/index/chroma/
```

Run:

```bash
python -m app.rag.index
```

---

## 🌐 Step 3 — Run the API server

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl -s http://127.0.0.1:8000/health | jq
```

---

## ❓ Query the RAG endpoint

```bash
curl -sS -X POST "http://127.0.0.1:8000/rag/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How long do Pro users have to request a refund?",
    "top_k": 5
  }' | jq
```

---

## 🧪 Tests

Run all tests:

```bash
pytest -q
```

Run integration-only tests:

```bash
pytest -q -m integration
```

---

## 📊 Evaluation (RAG Quality Metrics)

This project includes a small evaluation harness that checks:

- citation correctness (`must_cite`)
- answer correctness (`must_contain`)
- refusal behavior (`must_say_idk`)
- answer rate, citation rate, IDK rate

Run:

```bash
python -m eval.run_eval
```

Example output:

```json
{
  "n": 6,
  "answer_rate": 1.0,
  "citation_rate": 1.0,
  "idk_rate": 0.17,
  "must_cite_hit_rate": 1.0,
  "contain_hit_rate": 1.0
}
```

Full results are written to:

```bash
eval/results.json
```

---

## 🛡️ Hallucination Mitigation

This service includes several simple but effective safety techniques:

- **Evidence gate**: if retrieval score is too low → return “I don’t know…”
- **Strict JSON generation**: model must output only valid JSON
- **Citations required** for non-IDK answers
- **IDK answers return no citations** (prevents fake grounding)
- Hybrid retrieval combines:
  - vector similarity search
  - BM25 keyword search
  - keyword overlap boosting
  - fused scoring + reranking

---

## 🐳 Docker (Recommended)

### Start everything (API + Ollama)

```bash
docker compose up --build
```

The API will run on:

- http://127.0.0.1:8000

Ollama will run on:

- http://127.0.0.1:11434

### First-time model pull (Gemma 3 12B)

The first time you run Docker, Ollama may not have the model yet. Run:

```bash
docker compose up -d ollama
docker exec -it $(docker compose ps -q ollama) ollama pull gemma3:12b
docker compose up --build rag-api
```

---

## 🧰 Makefile (Convenience Commands)

If you have the `Makefile`, you can run:

```bash
make install
make ingest
make index
make run
make test
make eval
```

---

## ✅ GitHub Actions CI

This repo includes a basic CI workflow that runs tests on every push/PR:

- `.github/workflows/ci.yml`

---

## ⚠️ Known Limitations

This is a portfolio prototype, not a full production system:

- No auth / rate limiting
- No streaming responses
- Small demo document set (AcmeAI policies)
- Chroma index stored locally on disk

---

## ✅ Next Improvements (Roadmap)

Possible upgrades to make this more production-ready:

- add streaming generation
- add caching for retrieval + embeddings
- expand eval dataset (20–50 questions)
- add citation precision metrics (avoid unnecessary citations)
- add reranking improvements (MMR, cross-encoder reranker)
- add auth + rate limiting
