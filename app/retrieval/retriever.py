from typing import List, Tuple
import numpy as np
from app.core.logger import logger
from app.retrieval.confidence import score_results, ScoredResult, extract_gloss
from app.domain.diagnostics import RetrievalDiagnostics, FailureReason


def retrieve_relevant_chunks(query: str, vectorstore, k: int = 3) -> Tuple[List[ScoredResult], RetrievalDiagnostics]:
    """
    Retrieve relevant chunks and return detailed diagnostics.
    
    Args:
        query: The search query
        vectorstore: FAISS vector store to search
        k: Number of results to return
        
    Returns:
        Tuple of (Scored Results, Diagnostics Object)
    """
    logger.info(f"Retrieving top-{k} chunks for query: {query}")

    # Use similarity_search_with_score to get L2 distances
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k)

    logger.info(f"Retrieved {len(results_with_scores)} chunks")
    
    # -------------------------------------------------------------------------
    # CALCULATE METRICS
    # -------------------------------------------------------------------------
    scores = [score for _, score in results_with_scores]
    chunks = [doc.page_content for doc, _ in results_with_scores]
    
    # 1. Top Score
    top_score = scores[0] if scores else 0.0
    
    # 2. Score Delta (Top2 - Top1)
    # Larger delta = clearer winner (lower is better in L2)
    score_delta = 0.0
    if len(scores) >= 2:
        score_delta = scores[1] - scores[0]
        
    # 3. Result Density (Std Dev)
    result_density = 0.0
    if len(scores) > 1:
        result_density = float(np.std(scores))
        
    # 4. Gloss Diversity
    glosses = [extract_gloss(c) for c in chunks]
    unique_glosses = set(g for g in glosses if g)
    gloss_diversity = len(unique_glosses)
    
    # Build Diagnostics Object
    diagnostics = RetrievalDiagnostics(
        top_score=top_score,
        score_delta=score_delta,
        result_density=result_density,
        gloss_diversity=gloss_diversity,
        failure_reason=FailureReason.NONE # Default, will be updated by classifier
    )
    
    # Score results with confidence heuristics
    scored_results = score_results(results_with_scores)
    
    if scored_results:
        logger.info(f"Confidence: {scored_results[0].confidence.value} | Top Score: {top_score:.3f} | Delta: {score_delta:.3f}")
    
    return scored_results, diagnostics
