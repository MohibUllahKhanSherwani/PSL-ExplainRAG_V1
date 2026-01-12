import argparse
from pathlib import Path
from app.ingestion.loader import load_psl_glosses
from app.ingestion.chunker import glosses_to_chunks
from app.embeddings.embedder import get_embedding_model
from app.vectorstore.faiss_store import build_faiss_index
from app.retrieval.retriever import retrieve_relevant_chunks
from app.explanation.explainer import ExplanationEngine
from app.rendering.llm_renderer import LLMRenderer
from app.core.logger import logger


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PSL-ExplainRAG Query Pipeline")
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Enable LLM rendering (requires Ollama with llama3.2:1b)"
    )
    parser.add_argument(
        "--llm-model",
        default="llama3.2:1b",
        help="Ollama model to use for rendering (default: llama3.2:1b)"
    )
    args = parser.parse_args()

    # Load and preprocess PSL data
    glosses = load_psl_glosses(Path("data/raw/psl_glosses.json"))
    chunks = glosses_to_chunks(glosses)

    # Load embedding model
    embedding_model = get_embedding_model()

    # Build vector store
    vectorstore = build_faiss_index(chunks, embedding_model)
    
    # Initialize explanation engine (deterministic)
    explainer = ExplanationEngine()
    
    # Initialize LLM renderer (optional)
    llm_renderer = None
    if args.use_llm:
        llm_renderer = LLMRenderer(model=args.llm_model, enabled=True)
        if llm_renderer.is_available():
            logger.info(f"LLM rendering enabled with model: {args.llm_model}")
        else:
            logger.warning("LLM requested but not available - using templates only")
            llm_renderer = None

    # Test queries to demonstrate explanation generation
    test_queries = [
        "What does RUN mean in PSL?",           # Direct match, within-gloss ambiguity
        "How do I greet someone?",               # Direct match (HELLO)
        "I need assistance",                     # MEDIUM confidence
        "What is the weather like?",             # LOW confidence - refused
    ]

    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"QUERY: {query}")
        logger.info(f"{'='*60}")
        
        # Retrieve relevant chunks
        results = retrieve_relevant_chunks(query, vectorstore)
        
        # Generate explanation (deterministic)
        explanation = explainer.generate_explanation(query, results)
        
        # Display structured output
        logger.info(f"Answer Type: {explanation.answer_type.upper()}")
        logger.info(f"Confidence: {explanation.confidence.value}")
        logger.info(f"Primary Gloss: {explanation.primary_gloss}")
        logger.info(f"Has Ambiguity: {explanation.has_ambiguity} ({explanation.ambiguity_type})")
        
        # Display response
        logger.info(f"\n--- TEMPLATE RESPONSE ---\n{explanation.summary}")
        
        # Optional LLM rendering
        if llm_renderer:
            llm_response = llm_renderer.render_safe(explanation)
            logger.info(f"\n--- LLM RESPONSE ---\n{llm_response}")


if __name__ == "__main__":
    main()
