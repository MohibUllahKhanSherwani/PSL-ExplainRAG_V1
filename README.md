# PSL-ExplainRAG: Explorer RAG for Pakistan Sign Language

> [!NOTE]
> **🎓 Final Year Project (FYP) Self-Help Tool**
> This project was developed as a personal research and exploratory tool to support my Final Year Project (FYP) on sign language translation. It was designed to help reason about the technical nuances of disambiguating Pakistan Sign Language (PSL).

PSL-ExplainRAG is a **self-built, exploratory RAG-based knowledge tool** created to help reason about ambiguity and contextual meaning in **Pakistan Sign Language (PSL)** while working on a sign language translation system.

The project focuses on **structuring PSL linguistic knowledge**, embedding it locally, and retrieving relevant contextual explanations using semantic search. It is designed as a **personal research and learning tool**, not a production system.

---

## Project Motivation

During work on a PSL-based sign language translation system, it became clear that many PSL glosses are **ambiguous without context**.  
This project helps explore how **retrieval-augmented approaches** can ground and explain such ambiguity using structured linguistic knowledge.

---

## Current Capabilities

- **Knowledge Ingestion**: Defined a PSL domain schema and implemented ingestion of structured gloss data into semantic text chunks.
- **Embeddings & Retrieval**: Local semantic embeddings with sentence-transformers and FAISS vector storage for fast, local similarity search.
- **Confidence Scoring**: Deterministic heuristics (HIGH/MEDIUM/LOW) to detect ambiguity and ensure reliable retrieval.
- **Explanation Engine**: Template-based synthesis of grounding context without requiring an LLM for core reasoning.
- **LLM Rendering Layer**: Optional, local LLM integration via Ollama for natural language formatting with strict guardrails.
- **Diagnostics & Telemetry**: Failure taxonomy and metrics (score delta, density) to improve retrieval quality.
- **Service Layer**: Thin FastAPI shell for querying the system with Pydantic-validated response formats.
- **Evaluation Framework**: Curated evaluation sets for measuring retrieval accuracy and ambiguity detection.

---

## Project Structure

```
PSL-ExplainRAG/
│
├── app/
│   ├── bridge/        # LangChain wrappers
│   ├── core/          # Logging and core utilities
│   ├── domain/        # PSL domain schema + diagnostics
│   ├── ingestion/     # Data loading and chunking
│   ├── embeddings/    # Local embedding model
│   ├── vectorstore/   # FAISS vector index
│   ├── retrieval/     # Retrieval + confidence scoring
│   ├── explanation/   # Template-based explanation synthesis
│   └── rendering/     # LLM rendering 
│
├── data/
│   └── raw/           # PSL gloss knowledge (19 glosses)
│
├── scripts/
│   ├── ingest_psl_data.py
│   ├── build_and_query_index.py
│   ├── test_langchain.py
│
├── eval/              # Evaluation Framework
│   ├── queries.json
│   └── run_eval.py
│
├── main.py            # FastAPI service
├── requirements.txt
└── README.md
```

---

## Requirements

- **Python 3.11** (recommended) — Python 3.13 has compatibility issues with PyTorch
- **Visual C++ Redistributable** (Windows) — Required for PyTorch DLLs
  - Download: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Optional: Ollama (for LLM Rendering)

If you want to use the `--use-llm` flag for natural language rendering:

1. **Install Ollama**: Download from https://ollama.com/download (~1.2GB)
2. **Pull the model**:
   ```bash
   ollama pull llama3.2:1b
   ```

---

## How to Run the Project (I personally used CMD to run this project and activate environment)

### 1. Clone the Repository
```bash
git clone https://github.com/MohibUllahKhanSherwani/PSL-ExplainRAG_V1.git
cd PSL-ExplainRAG_V1
```

### 2. Create and Activate a Virtual Environment

**Windows (Command Prompt):**
```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate
```

**macOS/Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Full Pipeline (CLI Mode)
```bash
# Default mode (deterministic templates, no LLM required)
python -m scripts.build_and_query_index

# With LLM rendering (requires Ollama installed)
python -m scripts.build_and_query_index --use-llm
```

### 5. Run as API Service (FastAPI)
```cmd
:: Start the server
uvicorn main:app --reload
```

---

## Tech Stack
- **Python 3.11**
- **FastAPI** (REST API layer)
- **LangChain** (orchestration, text splitting)
- **Sentence-Transformers** (local embeddings with `all-MiniLM-L6-v2`)
- **FAISS** (local vector similarity search)
- **Pydantic** (request/response validation)
- **Loguru** (structured logging)
- **Ollama** (optional, local LLM for natural language rendering)
- **NumPy** (metrics & density calculation)
