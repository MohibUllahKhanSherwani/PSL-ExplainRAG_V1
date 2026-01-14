import sys
import os
from pathlib import Path

# Ensure app modules are visible
sys.path.append(os.getcwd())

from app.vectorstore.faiss_store import build_faiss_index
from app.ingestion.loader import load_psl_glosses
from app.ingestion.chunker import glosses_to_chunks
from app.embeddings.embedder import get_embedding_model
from app.bridge.pipeline_chain import create_psl_chain

def test_chain():
    print("Initializing PSL-ExplainRAG (LangChain Mode)...")
    
    # Setup data (same as main script)
    glosses = load_psl_glosses(Path("data/raw/psl_glosses.json"))
    chunks = glosses_to_chunks(glosses)
    
    # Load embedding model and build vectorstore
    embedding_model = get_embedding_model()
    vectorstore = build_faiss_index(chunks, embedding_model)
    
    # Create Chain (Without LLM for this test, or purely template based)
    # We test the deterministic path to ensure logic preservation.
    chain = create_psl_chain(vectorstore=vectorstore, use_llm=False)
    
    print("\n=== Test 1: High Confidence Query (YES) ===")
    query1 = "How do I say YES?"
    response1 = chain.invoke({"query": query1})
    print(f"Query: {query1}")
    print(f"Response:\n{response1}")
    
    print("\n=== Test 2: OOD Query (Refusal Logic) ===")
    query2 = "What is the weather like?"
    response2 = chain.invoke({"query": query2})
    print(f"Query: {query2}")
    print(f"Response:\n{response2}")
    
    # Validation
    if "agree" in response1.lower() or "affirmative" in response1.lower():
        print("\n[PASS] Test 1: Retrieved newly added YES gloss.")
    else:
        print("\n[FAIL] Test 1: Did not retrieve YES gloss.")

    if "low quality" in response2.lower() or "out of domain" in response2.lower():
         print("[PASS] Test 2: Correctly refused OOD query.")
    else:
         print("[FAIL] Test 2: Failed to refuse OOD query.")

if __name__ == "__main__":
    test_chain()
