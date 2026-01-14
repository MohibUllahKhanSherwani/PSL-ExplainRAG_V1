from operator import itemgetter
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from app.explanation.explainer import ExplanationEngine, Explanation
from app.rendering.llm_renderer import LLMRenderer
from app.bridge.langchain_retriever import PSLRetriever
from app.domain.diagnostics import RetrievalDiagnostics, FailureReason
from app.retrieval.confidence import ScoredResult, ConfidenceLevel

def rebuild_objects_from_docs(docs: list) -> tuple:
    """
    Reconstruct internal PSL objects (ScoredResult, RetrievalDiagnostics) 
    from LangChain Documents to feed into the Explainer.
    """
    if not docs:
        # Construct empty/failure diagnostics if no docs found
        diag = RetrievalDiagnostics(
            top_score=0.0, score_delta=0.0, result_density=0.0, gloss_diversity=0,
            failure_reason=FailureReason.NO_MATCHES
        )
        return [], diag

    # Reconstruct Diagnostics from the first doc's metadata (broadcasted)
    meta = docs[0].metadata
    diagnostics = RetrievalDiagnostics(
        top_score=meta.get("top_score", 0.0),
        score_delta=meta.get("score_delta", 0.0),
        result_density=meta.get("result_density", 0.0),
        gloss_diversity=meta.get("gloss_diversity", 0),
        failure_reason=FailureReason(meta.get("failure_reason", "NONE"))
    )

    # Reconstruct ScoredResults
    results = []
    for d in docs:
        results.append(ScoredResult(
            content=d.page_content,
            score=d.metadata.get("score", 0.0),
            confidence=ConfidenceLevel(d.metadata.get("confidence", "LOW"))
        ))
        
    return results, diagnostics

def create_psl_chain(vectorstore, use_llm: bool = False):
    """
    Compiles the PSL RAG pipeline into a LangChain Runnable.
    
    Flow:
    1. Input (query) -> PSLRetriever
    2. Docs -> Reconstruct Objects (RunnableLambda)
    3. Objects -> Explainer (RunnableLambda)
    4. Explanation -> Renderer (RunnableLambda)
    """
    
    retriever = PSLRetriever(vectorstore=vectorstore)
    explainer = ExplanationEngine()
    llm_renderer = LLMRenderer() if use_llm else None
    
    # --- Step 2: Explanation Logic Wrapper ---
    def run_explainer(inputs: Dict[str, Any]) -> Explanation:
        query = inputs["query"]
        docs = inputs["docs"]
        results, diagnostics = rebuild_objects_from_docs(docs)
        return explainer.generate_explanation(query, results, diagnostics)

    # --- Step 3: Renderer Wrapper ---
    def run_renderer(explanation: Explanation) -> str:
        # Day 6 Requirement: Use PromptTemplate for the LLM instruction
        # even if we pass it to our custom renderer.
        
        # Check explicit failure bypass (Day 5 Agreement)
        terminal_failures = [
            FailureReason.NO_MATCHES, 
            FailureReason.DATA_INCOMPLETE, 
            FailureReason.POOR_QUALITY_MATCH
        ]
        
        if explanation.failure_reason in terminal_failures:
            return explanation.summary # Bypass logic
            
        if use_llm and llm_renderer and llm_renderer.is_available():
            # We strictly use the renderer's logic, but we could wrap the prompt here
            # to demonstrate LC usage. For now, we delegate to preserve the exact logic.
            return llm_renderer.render_safe(explanation)
        
        return explanation.summary

    # --- LCEL Chain Construction ---
    chain = (
        {
            "docs": itemgetter("query") | retriever,
            "query": itemgetter("query")
        }
        | RunnableLambda(run_explainer)
        | RunnableLambda(run_renderer)
    )
    
    return chain
