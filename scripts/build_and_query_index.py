from pathlib import Path
from app.ingestion.loader import load_psl_glosses
from app.ingestion.chunker import glosses_to_chunks
from app.embeddings.embedder import get_embedding_model
from app.vectorstore.faiss_store import build_faiss_index
from app.retrieval.retriever import retrieve_relevant_chunks
from app.explanation.explainer import ExplanationEngine
from app.core.logger import logger


def main():
    # Load and preprocess PSL data
    glosses = load_psl_glosses(Path("data/raw/psl_glosses.json"))
    chunks = glosses_to_chunks(glosses)

    # Load embedding model
    embedding_model = get_embedding_model()

    # Build vector store
    vectorstore = build_faiss_index(chunks, embedding_model)
    
    # Initialize explanation engine
    explainer = ExplanationEngine()

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
        
        # Generate explanation
        explanation = explainer.generate_explanation(query, results)
        
        # Display structured output
        logger.info(f"Answer Type: {explanation.answer_type.upper()}")
        logger.info(f"Confidence: {explanation.confidence.value}")
        logger.info(f"Primary Gloss: {explanation.primary_gloss}")
        logger.info(f"Has Ambiguity: {explanation.has_ambiguity} ({explanation.ambiguity_type})")
        logger.info(f"\n--- EXPLANATION ---\n{explanation.summary}")


if __name__ == "__main__":
    main()
