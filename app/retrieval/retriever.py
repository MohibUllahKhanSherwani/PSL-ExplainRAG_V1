from typing import List, Optional
from app.core.logger import logger
from app.retrieval.confidence import score_results, ScoredResult, ConfidenceLevel


def retrieve_relevant_chunks(query: str, vectorstore, k: int = 3) -> List[ScoredResult]:
    """
    Retrieve relevant chunks with similarity scores and confidence levels.
    
    Args:
        query: The search query
        vectorstore: FAISS vector store to search
        k: Number of results to return
        
    Returns:
        List of ScoredResult with content, score, and confidence level
    """
    logger.info(f"Retrieving top-{k} chunks for query: {query}")

    # Use similarity_search_with_score to get L2 distances
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k)

    logger.info(f"Retrieved {len(results_with_scores)} chunks")
    
    # Score results with confidence heuristics
    scored_results = score_results(results_with_scores)
    
    if scored_results:
        logger.info(f"Confidence level: {scored_results[0].confidence.value}")
        for i, result in enumerate(scored_results):
            logger.debug(f"  Result {i+1}: score={result.score:.3f}")
    
    return scored_results
