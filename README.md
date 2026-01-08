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
- Verified retrieval with natural-language questions (e.g. *“What does RUN mean in PSL?”*)

---

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
│   └── retrieval/     # Similarity-based retrieval
│
├── data/
│   └── raw/           # PSL gloss knowledge
│
├── scripts/
│   ├── ingest_psl_data.py
│   └── build_and_query_index.py
│
├── requirements.txt
└── README.md
```

---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/MohibUllahKhanSherwani/PSL-ExplainRAG_V1.git
cd PSL-ExplainRAG_V1
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Day 1: Ingestion & Chunking
```bash
python scripts/ingest_psl_data.py
```
This will:
- Load PSL gloss knowledge
- Generate semantic chunks
- Print a preview chunk to the console

### 5. Run Day 2: Build Vector Store & Query
```bash
python scripts/build_and_query_index.py
```
This will:
- Embed PSL semantic chunks locally
- Build a FAISS vector index
- Retrieve the most relevant PSL explanation for a sample query

#### Example Query
`What does RUN mean in PSL?`  
The system retrieves the most relevant PSL context based on semantic similarity.

---

## Tech Stack
- **Python**
- **LangChain**
- **Sentence-Transformers** (local embeddings)
- **FAISS** (local vector store)
- **Pydantic**
- **Loguru**

## Next Steps
- Add explanation generation on top of retrieved context
- Introduce confidence checks and refusal logic
- Expose retrieval and explanation via an API

## Notes
This project is intentionally kept local and reproducible to emphasize understanding of RAG fundamentals, system design, and applied AI reasoning.
