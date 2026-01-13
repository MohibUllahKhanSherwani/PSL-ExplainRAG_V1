from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List

# =============================================================================
# FAILURE TAXONOMY
# =============================================================================

class FailureReason(str, Enum):
    """
    Classification of why a retrieval request failed or was refused.
    Strictly prioritized:
    1. NO_MATCHES (Terminal)
    2. DATA_INCOMPLETE (Terminal)
    3. POOR_QUALITY_MATCH (Terminal - OOD)
    4. AMBIGUOUS_COLLISION (Recoverable - Needs disambiguation)
    5. NONE (Success)
    """
    NO_MATCHES = "NO_MATCHES"
    DATA_INCOMPLETE = "DATA_INCOMPLETE"
    POOR_QUALITY_MATCH = "POOR_QUALITY_MATCH"
    AMBIGUOUS_COLLISION = "AMBIGUOUS_COLLISION"
    NONE = "NONE"


# =============================================================================
# RETRIEVAL METRICS
# =============================================================================

class RetrievalDiagnostics(BaseModel):
    """
    Measurable signals from the retrieval layer.
    Used for debugging and failure classification.
    """
    top_score: float = Field(..., description="L2 distance of the best match (lower is better)")
    score_delta: float = Field(..., description="Top-2 score minus Top-1 score (gap)")
    result_density: float = Field(..., description="Standard deviation of top-K scores")
    gloss_diversity: int = Field(..., description="Count of unique glosses in top-K results")
    failure_reason: FailureReason = Field(default=FailureReason.NONE, description="Classified failure mode")
