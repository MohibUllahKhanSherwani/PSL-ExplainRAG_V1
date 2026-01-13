from enum import Enum
from typing import List, Tuple
from dataclasses import dataclass
import re
from app.core.logger import logger


class ConfidenceLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class ScoredResult:
    """A retrieval result with content, score, and confidence."""
    content: str
    score: float  # L2 distance (lower = more similar)
    confidence: ConfidenceLevel


# Configurable thresholds (tuned for MiniLM-L6-v2 model)
# Imported from centralized config to ensure consistency
from app.core.config import (
    SCORE_HIGH_THRESHOLD,
    OOD_L2_THRESHOLD as SCORE_LOW_THRESHOLD,
    AMBIGUITY_DELTA_THRESHOLD,
    AMBIGUITY_CLEAR_THRESHOLD
)


def extract_gloss(chunk: str) -> str:
    """Extract the PSL gloss name from a chunk's content."""
    match = re.search(r"PSL Gloss:\s*(\w+)", chunk)
    return match.group(1) if match else ""


def compute_confidence(
    scores: List[float],
    chunks: List[str]
) -> ConfidenceLevel:
    """
    Compute retrieval confidence using deterministic heuristics.
    
    Rules:
    1. Absolute threshold: score > 1.0 → LOW
    2. Delta check: small gap between top results → ambiguous
    3. Agreement: different glosses in top results → reduce confidence
    """
    if not scores:
        return ConfidenceLevel.LOW
    
    best_score = scores[0]
    
    # Rule 1: Absolute threshold - if best match is too far, return LOW
    if best_score > SCORE_LOW_THRESHOLD:
        logger.debug(f"LOW confidence: best score {best_score:.3f} > {SCORE_LOW_THRESHOLD}")
        return ConfidenceLevel.LOW
    
    # Rule 2: Delta check (requires 2+ results)
    if len(scores) >= 2:
        delta = scores[1] - scores[0]
        # 3. Check for ambiguity (similar scores with different glosses)
        # Only check if scores are close AND we aren't already confident
        if delta < AMBIGUITY_DELTA_THRESHOLD and best_score > SCORE_HIGH_THRESHOLD:
            return ConfidenceLevel.MEDIUM
            
        # 4. Check for Clear Winner
        # If delta is large, we are very confident
        if delta > AMBIGUITY_CLEAR_THRESHOLD:
            return ConfidenceLevel.HIGH
    
    # Rule 3: Agreement check (requires 2+ chunks)
    if len(chunks) >= 2:
        glosses = [extract_gloss(c) for c in chunks]
        unique_glosses = set(g for g in glosses if g)  # Filter empty
        if len(unique_glosses) > 1 and best_score > SCORE_HIGH_THRESHOLD:
            logger.debug(f"MEDIUM confidence: disagreement among glosses {unique_glosses}")
            return ConfidenceLevel.MEDIUM
    
    # Final decision based on best score
    if best_score < SCORE_HIGH_THRESHOLD:
        logger.debug(f"HIGH confidence: best score {best_score:.3f} < {SCORE_HIGH_THRESHOLD}")
        return ConfidenceLevel.HIGH
    else:
        logger.debug(f"MEDIUM confidence: best score {best_score:.3f} in mid-range")
        return ConfidenceLevel.MEDIUM


def score_results(
    results_with_scores: List[Tuple[object, float]],
) -> List[ScoredResult]:
    """
    Convert FAISS results with scores to ScoredResult objects with confidence.
    
    Args:
        results_with_scores: List of (Document, score) tuples from FAISS
        
    Returns:
        List of ScoredResult with content, score, and confidence level
    """
    if not results_with_scores:
        return []
    
    chunks = [doc.page_content for doc, _ in results_with_scores]
    scores = [score for _, score in results_with_scores]
    
    # Compute overall confidence based on all results
    confidence = compute_confidence(scores, chunks)
    
    return [
        ScoredResult(
            content=doc.page_content,
            score=score,
            confidence=confidence
        )
        for doc, score in results_with_scores
    ]
