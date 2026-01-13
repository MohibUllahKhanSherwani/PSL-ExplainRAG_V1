
# Centralized Configuration for PSL-ExplainRAG
# Includes strict thresholds for retrieval confidence and failure classification.

# =============================================================================
# RETRIEVAL THRESHOLDS (Tuned for all-MiniLM-L6-v2)
# =============================================================================

# L2 Distance Thresholds (Lower is better)
# Best match score above this value = OUT OF DOMAIN (POOR QUALITY)
# NOTE: This value is embedding-model dependent.
OOD_L2_THRESHOLD = 1.4

# High Confidence Threshold
# Best match score below this value = HIGH_CONFIDENCE
SCORE_HIGH_THRESHOLD = 0.9

# Ambiguity Detection
# If (Top2 - Top1) < this_delta, we have a collision
AMBIGUITY_DELTA_THRESHOLD = 0.1

# If (Top2 - Top1) > this_delta, we have a clear winner
AMBIGUITY_CLEAR_THRESHOLD = 0.3
