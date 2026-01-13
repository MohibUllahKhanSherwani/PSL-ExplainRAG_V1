from dataclasses import dataclass, field
from typing import List, Optional, Literal
import re
from app.retrieval.confidence import ScoredResult, ConfidenceLevel
from app.core.logger import logger
from app.domain.diagnostics import RetrievalDiagnostics, FailureReason
from app.retrieval.failure import classify_failure


@dataclass
class ParsedGloss:
    """Structured representation of a PSL gloss extracted from a chunk."""
    gloss: str
    meanings: List[str]
    context_notes: str
    examples: List[str]


@dataclass
class Explanation:
    """
    Structured explanation output designed for both direct use and LLM integration.
    """
    # Core response metadata
    query: str
    answer_type: Literal["direct", "tentative", "refused"]
    confidence: ConfidenceLevel
    
    # Failure Awareness (Day 5)
    failure_reason: FailureReason = FailureReason.NONE
    diagnostics: Optional[RetrievalDiagnostics] = None
    
    # Primary gloss information (if found)
    primary_gloss: Optional[str] = None
    primary_meanings: List[str] = field(default_factory=list)
    
    # Disambiguation information
    has_ambiguity: bool = False
    ambiguity_type: Optional[Literal["within_gloss", "across_results"]] = None
    disambiguation_hint: Optional[str] = None
    
    # Context for grounding
    context_notes: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    
    # Raw data for LLM integration
    raw_chunks: List[str] = field(default_factory=list)
    retrieval_scores: List[float] = field(default_factory=list)
    
    # Human-readable output
    summary: str = ""


class ExplanationEngine:
    """
    Deterministic explanation synthesizer that converts retrieved chunks
    into structured, grounded responses without using an LLM.
    """
    
    def parse_chunk(self, chunk_text: str) -> Optional[ParsedGloss]:
        """Extract structured data from a chunk's text content."""
        try:
            # Extract gloss name
            gloss_match = re.search(r"PSL Gloss:\s*(\w+)", chunk_text)
            gloss = gloss_match.group(1) if gloss_match else ""
            
            # Extract meanings
            meanings_match = re.search(r"Possible Meanings:\s*(.+?)(?:\n|$)", chunk_text)
            meanings_str = meanings_match.group(1) if meanings_match else ""
            meanings = [m.strip() for m in meanings_str.split(",") if m.strip()]
            
            # Extract context notes
            context_match = re.search(r"Context Notes:\s*(.+?)(?:\n|$)", chunk_text)
            context_notes = context_match.group(1).strip() if context_match else ""
            
            # Extract examples
            examples_match = re.search(r"Usage Examples:\s*(.+?)(?:\n|$)", chunk_text)
            examples_str = examples_match.group(1) if examples_match else ""
            examples = [e.strip() for e in examples_str.split("|") if e.strip()]
            
            if not gloss:
                return None
                
            return ParsedGloss(
                gloss=gloss,
                meanings=meanings,
                context_notes=context_notes,
                examples=examples
            )
        except Exception as e:
            logger.warning(f"Failed to parse chunk: {e}")
            return None
    
    def analyze_ambiguity(
        self, 
        parsed_glosses: List[ParsedGloss]
    ) -> tuple[bool, Optional[Literal["within_gloss", "across_results"]], Optional[str]]:
        """
        Detect ambiguity in the retrieved results.
        
        Returns:
            (has_ambiguity, ambiguity_type, disambiguation_hint)
        """
        if not parsed_glosses:
            return False, None, None
        
        primary = parsed_glosses[0]
        
        # Check within-gloss ambiguity (multiple meanings)
        if len(primary.meanings) > 1:
            hint = f"Consider the context: {primary.context_notes}"
            return True, "within_gloss", hint
        
        # Check across-results ambiguity (different glosses in top results)
        if len(parsed_glosses) > 1:
            unique_glosses = set(g.gloss for g in parsed_glosses)
            if len(unique_glosses) > 1:
                other_glosses = [g.gloss for g in parsed_glosses[1:] if g.gloss != primary.gloss]
                hint = f"Other possibly relevant signs: {', '.join(other_glosses)}"
                return True, "across_results", hint
        
        return False, None, None
    
    def select_strategy(
        self, 
        confidence: ConfidenceLevel
    ) -> Literal["direct", "tentative", "refused"]:
        """Select response strategy based on confidence level."""
        if confidence == ConfidenceLevel.HIGH:
            return "direct"
        elif confidence == ConfidenceLevel.MEDIUM:
            return "tentative"
        else:
            return "refused"
    
    def render_summary(
        self, 
        answer_type: Literal["direct", "tentative", "refused"],
        parsed: Optional[ParsedGloss],
        has_ambiguity: bool,
        disambiguation_hint: Optional[str],
        failure_reason: FailureReason = FailureReason.NONE
    ) -> str:
        """Generate human-readable summary using templates."""
        
        if answer_type == "refused":
            # Specialized failure messages based on FailureReason
            if failure_reason == FailureReason.POOR_QUALITY_MATCH:
                return (
                    "I don't have reliable information for this query.\n"
                    "The retrieved matches were low quality (Out of Domain).\n"
                    "Consider rephrasing or asking about a specific PSL sign."
                )
            elif failure_reason == FailureReason.NO_MATCHES:
                return "No matching PSL signs were found in the knowledge base."
            
            if parsed:
                return (
                    f"I don't have reliable information for this query.\n"
                    f"The closest match was \"{parsed.gloss}\" but confidence is too low.\n"
                    f"Consider rephrasing or checking if this concept exists in PSL."
                )
            return (
                "I don't have reliable information for this query.\n"
                "No reliably matching PSL signs were found."
            )
        
        if not parsed:
            return "No matching information found."
        
        # Build the main answer
        meanings_str = ", ".join(parsed.meanings)
        examples_str = " | ".join(parsed.examples[:2])  # Limit to 2 examples
        
        if answer_type == "direct":
            summary = (
                f"The PSL sign \"{parsed.gloss}\" can mean: {meanings_str}.\n"
                f"{parsed.context_notes}\n"
                f"Examples: {examples_str}"
            )
        else:  # tentative
            summary = (
                f"Based on available knowledge, \"{parsed.gloss}\" may be relevant.\n"
                f"Possible meanings: {meanings_str}.\n"
                f"Note: This is a tentative match. {parsed.context_notes}"
            )
        
        # Append ambiguity note if applicable
        if has_ambiguity and disambiguation_hint:
            summary += f"\n\n Ambiguity detected: {disambiguation_hint}"
        
        return summary
    
    def generate_explanation(
        self, 
        query: str, 
        results: List[ScoredResult],
        diagnostics: RetrievalDiagnostics
    ) -> Explanation:
        """
        Main entry point: Generate a structured explanation from retrieval results.
        Includes failure classification handling.
        """
        raw_chunks = [r.content for r in results]
        
        # 1. Deterministic Failure Classification
        classified_failure = classify_failure(diagnostics, raw_chunks)
        diagnostics.failure_reason = classified_failure  # Stamp logic onto diagnostics
        
        # 2. Handle Terminal Failures Immediately
        if classified_failure in [FailureReason.NO_MATCHES, FailureReason.DATA_INCOMPLETE, FailureReason.POOR_QUALITY_MATCH]:
             return Explanation(
                query=query,
                answer_type="refused",
                confidence=ConfidenceLevel.LOW,
                failure_reason=classified_failure,
                diagnostics=diagnostics,
                summary=self.render_summary("refused", None, False, None, classified_failure)
            )

        # Get confidence from first result
        confidence = results[0].confidence
        
        # Parse all chunks
        parsed_glosses = []
        for result in results:
            parsed = self.parse_chunk(result.content)
            if parsed:
                parsed_glosses.append(parsed)
        
        # Analyze ambiguity
        has_ambiguity, ambiguity_type, disambiguation_hint = self.analyze_ambiguity(parsed_glosses)
        
        # Select response strategy
        answer_type = self.select_strategy(confidence)
        
        # Get primary gloss info
        primary = parsed_glosses[0] if parsed_glosses else None
        
        # Generate summary
        summary = self.render_summary(answer_type, primary, has_ambiguity, disambiguation_hint)
        
        # Build explanation object
        return Explanation(
            query=query,
            answer_type=answer_type,
            confidence=confidence,
            failure_reason=classified_failure,
            diagnostics=diagnostics,
            primary_gloss=primary.gloss if primary else None,
            primary_meanings=primary.meanings if primary else [],
            has_ambiguity=has_ambiguity,
            ambiguity_type=ambiguity_type,
            disambiguation_hint=disambiguation_hint,
            context_notes=primary.context_notes if primary else None,
            examples=primary.examples if primary else [],
            raw_chunks=[r.content for r in results],
            retrieval_scores=[r.score for r in results],
            summary=summary
        )
