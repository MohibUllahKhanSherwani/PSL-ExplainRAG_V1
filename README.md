# PSL-ExplainRAG

PSL-ExplainRAG is a **self-built, exploratory RAG-based knowledge tool** created to help reason about ambiguity and contextual meaning in **Pakistan Sign Language (PSL)** while working on a sign language translation system.

The project focuses on **structuring PSL linguistic knowledge**, embedding it locally, and retrieving relevant contextual explanations using semantic search. It is designed as a **personal research and learning tool**, not a production system.

---

## Project Motivation

During work on a PSL-based sign language translation system, it became clear that many PSL glosses are **ambiguous without context**.  
This project helps explore how **retrieval-augmented approaches** can ground and explain such ambiguity using structured linguistic knowledge.

---

## Current Capabilities

### Day 1 – Knowledge Ingestion & Chunking
- Defined a **PSL domain schema** for gloss-level linguistic knowledge
- Implemented ingestion of PSL gloss data
- Converted structured gloss entries into **semantic text chunks**
- Applied **recursive text splitting** to preserve contextual meaning

### Day 2 – Embeddings & Retrieval
- Generated **local semantic embeddings** using a sentence-transformer model
- Built a **local FAISS vector store** for PSL knowledge
- Implemented **similarity-based retrieval** for PSL-related queries
- Verified retrieval with natural-language questions (e.g. *"What does RUN mean in PSL?"*)

### Day 3 – Confidence Scoring & Explanation Layer
- Implemented **deterministic confidence heuristics** (HIGH / MEDIUM / LOW)
  - Absolute score thresholds
  - Score delta analysis for ambiguity detection
  - Agreement checking across retrieved chunks
- Built a **template-based explanation engine** (no LLM required)
  - **Direct** answers for HIGH confidence matches
  - **Tentative** answers for MEDIUM confidence matches
  - **Refusal** responses for LOW confidence (avoids hallucination)
- Added **ambiguity detection** (within-gloss and across-results)
- Designed **LLM-ready output structure** for future integration

### Day 4 – LLM Renderer Layer (Optional)
- Added **optional LLM rendering** using Ollama (local, no API)
  - Enabled via `--use-llm` flag (defaults to deterministic templates)
  - LLM acts purely as **text formatter** — no reasoning
- Implemented **strict renderer prompt** (non-expert role)
- Added **guardrail validation** (length + content checks)
  - Blocks hallucinated terms and answer patterns in refused responses
- System fully functions without LLM using template summaries


## Project Structure

```
PSL-ExplainRAG/
│
├── app/
│   ├── core/          # Logging and core utilities
│   ├── domain/        # PSL domain schema
│   ├── ingestion/     # Data loading and chunking
│   ├── embeddings/    # Local embedding model
│   ├── vectorstore/   # FAISS vector index
│   ├── retrieval/     # Similarity-based retrieval + confidence scoring
│   ├── explanation/   # Template-based explanation synthesis
│   └── rendering/     # Optional LLM rendering (Day 4)
│
├── data/
│   └── raw/           # PSL gloss knowledge (8 glosses)
│
├── scripts/
│   ├── ingest_psl_data.py
│   └── build_and_query_index.py
│
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
3. **Recommended model**: `llama3.2:1b` (~1.3GB) — fast, good for rendering-only tasks

> **Note**: The system works 100% without Ollama using deterministic templates.

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

### 4. Run the Full Pipeline
```bash
# Default mode (deterministic templates, no LLM required)
python -m scripts.build_and_query_index

# With LLM rendering (requires Ollama installed)
python -m scripts.build_and_query_index --use-llm
```

This will:
- Load PSL gloss knowledge (8 entries)
- Generate semantic chunks
- Build a FAISS vector index
- Run test queries with confidence scoring
- Generate grounded explanations
- (Optional) Render via LLM if `--use-llm` flag is set

---

## Example Output

```
QUERY: What does RUN mean in PSL?
============================================================
Answer Type: DIRECT
Confidence: HIGH
Primary Gloss: RUN
Has Ambiguity: True (within_gloss)

--- EXPLANATION ---
The PSL sign "RUN" can mean: run, operate, flow.
Meaning depends on whether the subject is a human, a machine, or a liquid.
Examples: He runs every morning | The engine is running

Ambiguity detected: Consider the context: Meaning depends on whether the 
subject is a human, a machine, or a liquid.
```

---

## Confidence Levels

| Level | Score Range | Response |
|-------|-------------|----------|
| **HIGH** | < 0.9 | Direct answer with full context |
| **MEDIUM** | 0.9 - 1.4 | Tentative answer with caveats |
| **LOW** | > 1.4 | Refusal — avoids hallucination |

---

## Tech Stack
- **Python 3.11**
- **LangChain** (text splitting, vector store integration)
- **Sentence-Transformers** (local embeddings with `all-MiniLM-L6-v2`)
- **FAISS** (local vector similarity search)
- **Pydantic** (data validation)
- **Loguru** (structured logging)
- **Ollama** (optional, local LLM for natural language rendering)

---

## Next Steps
- [ ] Persist FAISS index to avoid rebuilding on each run
- [ ] Expose retrieval and explanation via a FastAPI endpoint
- [x] ~~Integrate LLM for natural language explanation generation~~ ✅ Done (Day 4)
- [ ] Add more PSL glosses to the knowledge base

---

## Notes
This project is intentionally kept local and reproducible to emphasize understanding of RAG fundamentals, system design, and applied AI reasoning.
