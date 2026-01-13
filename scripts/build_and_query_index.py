import argparse
from pathlib import Path
from app.ingestion.loader import load_psl_glosses
from app.ingestion.chunker import glosses_to_chunks
from app.embeddings.embedder import get_embedding_model
from app.vectorstore.faiss_store import build_faiss_index
from app.retrieval.retriever import retrieve_relevant_chunks
from app.explanation.explainer import ExplanationEngine
from app.rendering.llm_renderer import LLMRenderer
from app.domain.diagnostics import RetrievalDiagnostics, FailureReason
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
        "What does RUN mean in PSL?",            # HIGH confidence - direct
        "How do I say YES?",                     # HIGH confidence - new gloss
        "What is the sign for SCHOOL?",          # HIGH confidence - new gloss
        "I am feeling HAPPY",                    # HIGH confidence - new gloss
        "What is the weather like?",             # LOW confidence - refused
    ]

    # Process each query
    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"QUERY: {query}")
        logger.info(f"{'='*60}")
        
        # 3. Retrieve relevant chunks (returns results + diagnostics)
        results, diagnostics = retrieve_relevant_chunks(query, vectorstore, k=3)
        
        # 4. Generate deterministic explanation
        explanation = explainer.generate_explanation(query, results, diagnostics)
        
        # Log Diagnostics (Day 5 Measurement)
        if diagnostics:
            logger.info(f"Diagnostics | Top Score: {diagnostics.top_score:.3f} | Delta: {diagnostics.score_delta:.3f} | Density: {diagnostics.result_density:.3f} | Diversity: {diagnostics.gloss_diversity}")
            if diagnostics.failure_reason != FailureReason.NONE:
                logger.warning(f"Failure Detected: {diagnostics.failure_reason.value}")
        
        # Display structured output
        logger.info(f"Answer Type: {explanation.answer_type.upper()}")
        logger.info(f"Confidence: {explanation.confidence.value}")
        logger.info(f"Primary Gloss: {explanation.primary_gloss}")
        logger.info(f"Has Ambiguity: {explanation.has_ambiguity} ({explanation.ambiguity_type})")

        # Display response
        logger.info(f"\n--- TEMPLATE RESPONSE ---\n{explanation.summary}\n")
        
        # 5. Optional LLM Rendering (Day 4 + Day 5 Safety Contract)
        if args.use_llm and llm_renderer and llm_renderer.is_available():
            # STRICT LLM FAILURE CONTRACT (Day 5)
            # Bypass LLM for terminal failures
            terminal_failures = [
                FailureReason.NO_MATCHES, 
                FailureReason.DATA_INCOMPLETE, 
                FailureReason.POOR_QUALITY_MATCH
            ]
            
            if explanation.failure_reason in terminal_failures:
                logger.info(f"LLM Bypassed due to terminal failure: {explanation.failure_reason.value}")
            else:
                llm_response = llm_renderer.render_safe(explanation)
                logger.info(f"\n--- LLM RESPONSE ---\n{llm_response}")
        
        logger.info("="*60)


if __name__ == "__main__":
    main()
