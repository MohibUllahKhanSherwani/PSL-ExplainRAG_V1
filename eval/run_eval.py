"""
Lightweight Evaluation Script for PSL-ExplainRAG (Day 8)

Runs gold queries against the existing retrieval pipeline and computes metrics.
No modifications to retrieval logic, thresholds, or embeddings.
"""
import sys
import os
import json
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

# Ensure app modules are visible
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.loader import load_psl_glosses
from app.ingestion.chunker import glosses_to_chunks
from app.embeddings.embedder import get_embedding_model
from app.vectorstore.faiss_store import build_faiss_index
from app.retrieval.retriever import retrieve_relevant_chunks
from app.explanation.explainer import ExplanationEngine
from app.domain.diagnostics import FailureReason


@dataclass
class EvalResult:
    """Single query evaluation result."""
    query_id: int
    query: str
    expected_gloss: Optional[str]
    expected_outcome: str
    actual_gloss: Optional[str]
    actual_outcome: str
    top_k_glosses: List[str]
    top_k_scores: List[float]
    score_delta: float
    failure_reason: str
    correct_retrieval: bool
    correct_outcome: bool


def extract_gloss_from_chunk(chunk: str) -> Optional[str]:
    """Extract gloss name from chunk text."""
    match = re.search(r"PSL Gloss:\s*(\w+)", chunk)
    return match.group(1) if match else None


def run_evaluation(queries_path: Path, k: int = 3) -> Dict[str, Any]:
    """
    Run evaluation on gold query set.
    
    Returns:
        Dictionary with results and computed metrics.
    """
    # Load gold queries
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    print(f"Loaded {len(queries)} gold queries from {queries_path}")
    
    # Initialize pipeline (uses existing logic verbatim)
    print("Initializing pipeline...")
    glosses = load_psl_glosses(Path("data/raw/psl_glosses.json"))
    chunks = glosses_to_chunks(glosses)
    embedding_model = get_embedding_model()
    vectorstore = build_faiss_index(chunks, embedding_model)
    explainer = ExplanationEngine()
    
    print(f"Pipeline ready. Running evaluation...\n")
    print("=" * 80)
    
    results: List[EvalResult] = []
    
    for q in queries:
        query_id = q["id"]
        query_text = q["query"]
        expected_gloss = q.get("expected_gloss")
        expected_outcome = q["expected_outcome"]
        
        # Run retrieval (existing logic, no modifications)
        scored_results, diagnostics = retrieve_relevant_chunks(
            query_text, vectorstore, k=k
        )
        
        # Generate explanation (existing logic)
        explanation = explainer.generate_explanation(
            query_text, scored_results, diagnostics
        )
        
        # Extract actual results
        top_k_glosses = [
            extract_gloss_from_chunk(r.content) 
            for r in scored_results
        ]
        top_k_scores = [float(r.score) for r in scored_results]  # Convert to native float
        actual_gloss = top_k_glosses[0] if top_k_glosses else None
        
        # Determine actual outcome
        if explanation.answer_type == "refused":
            actual_outcome = "refusal"
        elif explanation.has_ambiguity:
            actual_outcome = "ambiguity"
        else:
            actual_outcome = "answer"
        
        # Correctness checks
        correct_retrieval = (
            expected_gloss is None or 
            (actual_gloss and actual_gloss.upper() == expected_gloss.upper())
        )
        correct_outcome = (actual_outcome == expected_outcome)
        
        result = EvalResult(
            query_id=query_id,
            query=query_text,
            expected_gloss=expected_gloss,
            expected_outcome=expected_outcome,
            actual_gloss=actual_gloss,
            actual_outcome=actual_outcome,
            top_k_glosses=top_k_glosses,
            top_k_scores=top_k_scores,
            score_delta=float(diagnostics.score_delta),  # Convert to native float
            failure_reason=diagnostics.failure_reason.value,
            correct_retrieval=correct_retrieval,
            correct_outcome=correct_outcome
        )
        results.append(result)
        
        # Log each result
        status = "PASS" if (correct_retrieval and correct_outcome) else "FAIL"
        print(f"[{status}] Q{query_id}: {query_text[:40]}...")
        print(f"    Expected: {expected_gloss or 'OOD'} -> {expected_outcome}")
        print(f"    Actual:   {actual_gloss or 'NONE'} -> {actual_outcome}")
        print(f"    Top-{k} Scores: {[f'{s:.3f}' for s in top_k_scores]}")
        print(f"    Score Delta: {diagnostics.score_delta:.3f}")
        print(f"    Failure Mode: {diagnostics.failure_reason.value}")
        print()
    
    print("=" * 80)
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    return {
        "results": [asdict(r) for r in results],
        "metrics": metrics
    }


def compute_metrics(results: List[EvalResult]) -> Dict[str, Any]:
    """Compute lightweight evaluation metrics."""
    total = len(results)
    
    # 1. Top-1 Retrieval Accuracy
    in_domain = [r for r in results if r.expected_gloss is not None]
    correct_retrieval = sum(1 for r in in_domain if r.correct_retrieval)
    retrieval_accuracy = correct_retrieval / len(in_domain) if in_domain else 0.0
    
    # 2. Ambiguity Detection Accuracy
    expected_ambiguous = [r for r in results if r.expected_outcome == "ambiguity"]
    correct_ambiguity = sum(1 for r in expected_ambiguous if r.actual_outcome == "ambiguity")
    ambiguity_accuracy = correct_ambiguity / len(expected_ambiguous) if expected_ambiguous else 0.0
    
    # 3. False Confident Answer Count
    # Cases where we gave a confident answer but should have refused or flagged ambiguity
    false_confident = sum(
        1 for r in results 
        if r.actual_outcome == "answer" and r.expected_outcome != "answer"
    )
    
    # 4. OOD Rejection Rate
    ood_queries = [r for r in results if r.expected_gloss is None]
    correct_rejections = sum(1 for r in ood_queries if r.actual_outcome == "refusal")
    ood_rejection_rate = correct_rejections / len(ood_queries) if ood_queries else 0.0
    
    # 5. Overall Outcome Accuracy
    correct_outcomes = sum(1 for r in results if r.correct_outcome)
    outcome_accuracy = correct_outcomes / total if total else 0.0
    
    metrics = {
        "total_queries": total,
        "top1_retrieval_accuracy": round(retrieval_accuracy, 3),
        "ambiguity_detection_accuracy": round(ambiguity_accuracy, 3),
        "false_confident_count": false_confident,
        "ood_rejection_rate": round(ood_rejection_rate, 3),
        "overall_outcome_accuracy": round(outcome_accuracy, 3),
        "breakdown": {
            "in_domain_queries": len(in_domain),
            "ood_queries": len(ood_queries),
            "expected_ambiguous": len(expected_ambiguous)
        }
    }
    
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    print(f"Total Queries:              {total}")
    print(f"Top-1 Retrieval Accuracy:   {retrieval_accuracy:.1%} ({correct_retrieval}/{len(in_domain)})")
    print(f"Ambiguity Detection:        {ambiguity_accuracy:.1%} ({correct_ambiguity}/{len(expected_ambiguous)})")
    print(f"False Confident Answers:    {false_confident}")
    print(f"OOD Rejection Rate:         {ood_rejection_rate:.1%} ({correct_rejections}/{len(ood_queries)})")
    print(f"Overall Outcome Accuracy:   {outcome_accuracy:.1%} ({correct_outcomes}/{total})")
    print("=" * 80)
    
    return metrics


def main():
    queries_path = Path(__file__).parent / "queries.json"
    
    if not queries_path.exists():
        print(f"Error: {queries_path} not found")
        sys.exit(1)
    
    output = run_evaluation(queries_path)
    
    # Save results to JSON
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
