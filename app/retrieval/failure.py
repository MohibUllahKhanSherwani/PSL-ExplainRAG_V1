from typing import List, Tuple
from app.domain.diagnostics import FailureReason, RetrievalDiagnostics
from app.core.config import OOD_L2_THRESHOLD, AMBIGUITY_DELTA_THRESHOLD
from app.core.logger import logger
import re

def classify_failure(
    diagnostics: RetrievalDiagnostics,
    raw_chunks: List[str]
) -> FailureReason:
    """
    Deterministic failure classifier.
    Strictly prioritized evaluation order:
    1. NO_MATCHES
    2. DATA_INCOMPLETE
    3. POOR_QUALITY_MATCH (OOD)
    4. AMBIGUOUS_COLLISION
    5. NONE (Success)
    """
    
    # 1. NO_MATCHES
    # Note: Typically handled before calling this, but safe to check
    if not raw_chunks:
        return FailureReason.NO_MATCHES
        
    # 2. DATA_INCOMPLETE
    # Check if the primary chunk is malformed (missing Gloss or Meanings)
    primary_chunk = raw_chunks[0]
    gloss_match = re.search(r"PSL Gloss:\s*(\w+)", primary_chunk)
    meanings_match = re.search(r"Possible Meanings:\s*(.+?)(?:\n|$)", primary_chunk)
    
    if not gloss_match or not meanings_match:
        logger.warning(f"Failure Detected: DATA_INCOMPLETE for chunk: {primary_chunk[:50]}...")
        return FailureReason.DATA_INCOMPLETE
    
    # 3. POOR_QUALITY_MATCH (OOD)
    if diagnostics.top_score > OOD_L2_THRESHOLD:
        logger.info(f"Failure Detected: OOD (Score {diagnostics.top_score:.3f} > {OOD_L2_THRESHOLD})")
        return FailureReason.POOR_QUALITY_MATCH
    
    # 4. AMBIGUOUS_COLLISION
    # Check if delta is small AND distinct glosses actally exist in top results
    if diagnostics.score_delta < AMBIGUITY_DELTA_THRESHOLD:
        # Verify valid collision (more than 1 distinct gloss)
        if diagnostics.gloss_diversity > 1:
            logger.info(f"Failure Detected: COLLISION (Delta {diagnostics.score_delta:.3f} < {AMBIGUITY_DELTA_THRESHOLD})")
            return FailureReason.AMBIGUOUS_COLLISION
            
    return FailureReason.NONE
