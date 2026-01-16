"""
FastAPI Minimal Service for PSL-ExplainRAG

Transport-only wrapper. Zero reasoning logic.
"""
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, model_validator
from fastapi import FastAPI

# --- Core Pipeline Imports (verbatim usage) ---
from app.ingestion.loader import load_psl_glosses
from app.ingestion.chunker import glosses_to_chunks
from app.embeddings.embedder import get_embedding_model
from app.vectorstore.faiss_store import build_faiss_index
from app.retrieval.retriever import retrieve_relevant_chunks
from app.explanation.explainer import ExplanationEngine


# =============================================================================
# PYDANTIC MODELS (Type Safety Only)
# =============================================================================

class QueryRequest(BaseModel):
    query: str


class AmbiguityDetail(BaseModel):
    candidates: List[str]


class QueryResponse(BaseModel):
    """
    Response model enforcing exactly one of: answer, ambiguity, or refusal.
    """
    answer: Optional[str] = None
    ambiguity: Optional[AmbiguityDetail] = None
    refusal: Optional[str] = None

    @model_validator(mode="after")
    def exactly_one_field(self):
        fields = [self.answer, self.ambiguity, self.refusal]
        count = sum(1 for f in fields if f is not None)
        if count != 1:
            raise ValueError("Exactly one of answer, ambiguity, or refusal must be set")
        return self


# =============================================================================
# APPLICATION INITIALIZATION
# =============================================================================

app = FastAPI(title="PSL-ExplainRAG", version="1.0.0")

# Load pipeline components once at startup
_glosses = load_psl_glosses(Path("data/raw/psl_glosses.json"))
_chunks = glosses_to_chunks(_glosses)
_embedding_model = get_embedding_model()
_vectorstore = build_faiss_index(_chunks, _embedding_model)
_explainer = ExplanationEngine()


# =============================================================================
# ENDPOINT
# =============================================================================

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    POST /query - Transport layer for the deterministic RAG pipeline.
    
    Maps pipeline output directly to response. No domain logic here.
    """
    # 1. Call existing pipeline verbatim
    results, diagnostics = retrieve_relevant_chunks(
        request.query, _vectorstore, k=3
    )
    explanation = _explainer.generate_explanation(
        request.query, results, diagnostics
    )
    
    # 2. Deterministic mapping (no parsing, no logic - just field assignment)
    if explanation.answer_type == "refused":
        return QueryResponse(refusal=explanation.summary)
    
    if explanation.has_ambiguity:
        # Candidates come directly from pipeline's parsed data
        candidates = []
        if explanation.primary_meanings:
            candidates = explanation.primary_meanings
        elif explanation.disambiguation_hint:
            # Extract gloss names from hint if meanings not available
            candidates = [explanation.primary_gloss or "unknown"]
        return QueryResponse(ambiguity=AmbiguityDetail(candidates=candidates))
    
    return QueryResponse(answer=explanation.summary)
