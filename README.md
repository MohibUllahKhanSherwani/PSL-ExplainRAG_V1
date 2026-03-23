<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PSL-ExplainRAG README</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #c9d1d9;
            background-color: #0d1117;
            padding: 45px;
            max-width: 900px;
            margin: auto;
        }
        h1, h2, h3 { color: #f0f6fc; border-bottom: 1px solid #30363d; padding-bottom: 0.3em; }
        .alert {
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid transparent;
            border-radius: 6px;
        }
        .alert-note {
            color: #f0f6fc;
            background-color: rgba(33, 38, 45, 0.4);
            border-color: #30363d;
            border-left: 5px solid #1f6feb;
        }
        code {
            font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
            background-color: rgba(110, 118, 129, 0.4);
            padding: 0.2em 0.4em;
            border-radius: 6px;
            font-size: 85%;
        }
        pre {
            background-color: #161b22;
            padding: 16px;
            border-radius: 6px;
            overflow: auto;
        }
        pre code { background-color: transparent; }
        hr { height: 0.25em; background-color: #30363d; border: 0; margin: 24px 0; }
        ul { padding-left: 20px; }
        li { margin-bottom: 8px; }
    </style>
</head>
<body>
    <h1>PSL-ExplainRAG: Explorer RAG for Pakistan Sign Language</h1>

    <div class="alert alert-note">
        <strong>🎓 Final Year Project (FYP) Self-Help Tool</strong><br>
        This project was developed as a personal research and exploratory tool to support my Final Year Project (FYP) on sign language translation. It was designed to help reason about the technical nuances of disambiguating Pakistan Sign Language (PSL).
    </div>

    <p>PSL-ExplainRAG is a <strong>self-built, exploratory RAG-based knowledge tool</strong> created to help reason about ambiguity and contextual meaning in <strong>Pakistan Sign Language (PSL)</strong>.</p>

    <hr>

    <h2>Project Motivation</h2>
    <p>During work on a PSL-based sign language translation system, it became clear that many PSL glosses are <strong>ambiguous without context</strong>. This project helps explore how retrieval-augmented approaches can ground and explain such ambiguity.</p>

    <h2>Current Capabilities</h2>
    <ul>
        <li><strong>Knowledge Ingestion:</strong> PSL domain schema and structured gloss-to-text chunking.</li>
        <li><strong>Embeddings & Retrieval:</strong> Local similarity search using <code>sentence-transformers</code> and <code>FAISS</code>.</li>
        <li><strong>Confidence Scoring:</strong> Heuristics for ambiguity detection and hallucination prevention.</li>
        <li><strong>Explanation Engine:</strong> Grounded explanation synthesis via deterministic templates.</li>
        <li><strong>LLM Layer:</strong> Optional local rendering via <code>Ollama</code> with strict guardrails.</li>
        <li><strong>Service & Eval:</strong> thin <code>FastAPI</code> layer and automated evaluation metrics.</li>
    </ul>

    <hr>

    <h2>Project Structure</h2>
    <pre><code>PSL-ExplainRAG/
│
├── app/            # Core logic and domain models
├── data/raw/       # PSL gloss knowledge data
├── scripts/        # Ingestion and build scripts
├── eval/           # Evaluation Framework
└── main.py         # FastAPI service layer</code></pre>

    <hr>

    <h2>Requirements</h2>
    <ul>
        <li>Python 3.11 (highly recommended)</li>
        <li>Visual C++ Redistributable (for PyTorch on Windows)</li>
        <li>Ollama (Optional, for LLM rendering)</li>
    </ul>

    <hr>

    <h2>How to Run</h2>
    <h3>1. Setup Environment</h3>
    <pre><code># Windows (Command Prompt)
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt</code></pre>

    <h3>2. Run Pipeline</h3>
    <pre><code># CLI Mode
python -m scripts.build_and_query_index

# API Mode
uvicorn main:app --reload</code></pre>

    <hr>

    <h2>Tech Stack</h2>
    <ul>
        <li>Python, FastAPI, LangChain</li>
        <li>Sentence-Transformers, FAISS, Pydantic</li>
        <li>Loguru, Ollama, NumPy</li>
    </ul>
</body>
</html>
