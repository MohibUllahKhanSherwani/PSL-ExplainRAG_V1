"""
LLM Renderer Layer for PSL-ExplainRAG

This module provides an OPTIONAL LLM-based natural language renderer.
The LLM acts purely as a text formatter - all reasoning (confidence,
ambiguity detection, refusal logic) is done by the deterministic engine.

The system fully functions without the LLM using template-based summaries.
"""

import re
from dataclasses import asdict
from typing import Optional, Literal, List
from pydantic import BaseModel, field_validator
from app.explanation.explainer import Explanation
from app.retrieval.confidence import ConfidenceLevel
from app.core.logger import logger


# =============================================================================
# INPUT SCHEMA - What the LLM sees (no internal data like raw_chunks/scores)
# =============================================================================

class RendererInput(BaseModel):
    """Strict schema passed to LLM - only factual fields, no internal data."""
    
    # Response control (pre-determined by deterministic engine)
    answer_type: Literal["direct", "tentative", "refused"]
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    
    # Gloss information (may be None for refused)
    gloss: Optional[str] = None
    meanings: List[str] = []
    context_notes: Optional[str] = None
    examples: List[str] = []
    
    # Ambiguity information (pre-computed)
    has_ambiguity: bool = False
    ambiguity_type: Optional[str] = None  # "within_gloss" or "across_results"
    disambiguation_hint: Optional[str] = None
    
    # Original query for context
    query: str


# =============================================================================
# OUTPUT VALIDATION - Guardrails against hallucination
# =============================================================================

class RendererOutput(BaseModel):
    """Validated LLM output with length constraints."""
    rendered_text: str
    
    @field_validator('rendered_text')
    @classmethod
    def validate_length(cls, v):
        if len(v) > 500:
            raise ValueError("Response too long (max 500 chars) - potential hallucination")
        if len(v) < 10:
            raise ValueError("Response too short (min 10 chars)")
        return v.strip()


class GuardrailError(Exception):
    """Raised when LLM output fails validation."""
    pass


def validate_no_hallucination(output: str, input_data: RendererInput) -> bool:
    """
    Check that output doesn't contain fabricated content.
    
    Rules:
    - Refused responses must not contain answer patterns
    - Any quoted terms should exist in the input meanings/gloss
    """
    # Rule 1: Refused responses must not give answers
    if input_data.answer_type == "refused":
        forbidden_patterns = [
            r'\bmeans?\b',
            r'\bcan mean\b',
            r'\brepresents?\b',
            r'\bthe sign for\b',
            r'\bis used for\b',
        ]
        for pattern in forbidden_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                logger.warning(f"Guardrail violation: refused response contains '{pattern}'")
                return False
    
    # Rule 2: Check for fabricated quoted terms (simple heuristic)
    quoted_terms = re.findall(r'"([^"]+)"', output)
    allowed_terms = set(input_data.meanings)
    if input_data.gloss:
        allowed_terms.add(input_data.gloss)
    
    for term in quoted_terms:
        # Allow if it's a substring of allowed terms or vice versa
        term_lower = term.lower()
        if not any(term_lower in allowed.lower() or allowed.lower() in term_lower 
                   for allowed in allowed_terms):
            logger.warning(f"Guardrail violation: unknown quoted term '{term}'")
            return False
    
    return True


# =============================================================================
# STRICT RENDERER PROMPT - LLM as non-expert text formatter
# =============================================================================

RENDERER_PROMPT = """SYSTEM:
You are a rendering assistant for a Pakistan Sign Language (PSL) knowledge system.
Your ONLY job is to rewrite structured data into clear, natural English.
You are NOT an expert. You are a text formatter. All reasoning has been done for you.

STRICT RULES:
1. You may ONLY use information provided in the INPUT below.
2. You must NEVER add meanings, glosses, or examples not in the input.
3. You must NEVER invent or infer information.
4. You must NEVER use phrases like "I think", "perhaps", "probably", or make guesses.
5. You must NEVER override the answer_type - if it says "refused", you REFUSE.

RESPONSE STYLE (based on answer_type in input):
- "direct": Confident, declarative statements. State the facts clearly.
- "tentative": Use hedging language like "may", "possibly", "based on available knowledge".
- "refused": Politely decline. Say you don't have reliable information. Do NOT provide any answer.

AMBIGUITY HANDLING:
- If has_ambiguity is true, explicitly mention that the sign has multiple interpretations.
- Include the disambiguation_hint if provided.

FORMAT:
- 2-4 sentences for direct/tentative answers
- 1-2 sentences for refused answers
- No bullet points, no markdown, just plain natural sentences

INPUT:
{input_json}

OUTPUT:
Write the natural language response now. Include ONLY information from the INPUT."""


# =============================================================================
# LLM RENDERER CLASS - Optional layer with toggle
# =============================================================================

class LLMRenderer:
    """
    Optional LLM-based natural language renderer.
    
    The LLM receives pre-computed facts and renders them into natural language.
    It does NOT perform any reasoning - confidence, ambiguity, and refusal
    are all determined by the deterministic ExplanationEngine.
    
    Usage:
        renderer = LLMRenderer()
        if renderer.is_available():
            result = renderer.render(explanation)
        else:
            result = explanation.summary  # Fallback to template
    """
    
    def __init__(
        self,
        model: str = "llama3.2:1b",
        temperature: float = 0.1,  # Low for deterministic output
        max_tokens: int = 200,
        enabled: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enabled = enabled
        self._ollama = None
        self._available: Optional[bool] = None
    
    def is_available(self) -> bool:
        """Check if Ollama is available and the model is loaded."""
        if not self.enabled:
            return False
        
        if self._available is not None:
            return self._available
        
        try:
            import ollama
            self._ollama = ollama
            # Check if model exists (will raise if not)
            ollama.show(self.model)
            self._available = True
            logger.info(f"LLM renderer available: {self.model}")
        except ImportError:
            logger.warning("ollama package not installed - LLM rendering disabled")
            self._available = False
        except Exception as e:
            logger.warning(f"Ollama not available: {e} - LLM rendering disabled")
            self._available = False
        
        return self._available
    
    def _explanation_to_input(self, explanation: Explanation) -> RendererInput:
        """Convert Explanation to RendererInput (filters out internal data)."""
        return RendererInput(
            answer_type=explanation.answer_type,
            confidence=explanation.confidence.value,
            gloss=explanation.primary_gloss,
            meanings=explanation.primary_meanings,
            context_notes=explanation.context_notes,
            examples=explanation.examples,
            has_ambiguity=explanation.has_ambiguity,
            ambiguity_type=explanation.ambiguity_type,
            disambiguation_hint=explanation.disambiguation_hint,
            query=explanation.query,
        )
    
    def render(self, explanation: Explanation) -> str:
        """
        Render an Explanation into natural language using the LLM.
        
        Args:
            explanation: Pre-computed Explanation from ExplanationEngine
            
        Returns:
            Natural language response string
            
        Raises:
            GuardrailError: If LLM output fails validation
            RuntimeError: If Ollama is not available
        """
        if not self.is_available():
            raise RuntimeError("LLM renderer not available - use explanation.summary instead")
        
        # Convert to input schema (filters internal data)
        input_data = self._explanation_to_input(explanation)
        input_json = input_data.model_dump_json(indent=2)
        
        # Build prompt
        prompt = RENDERER_PROMPT.format(input_json=input_json)
        
        logger.debug(f"LLM input: {input_json}")
        
        # Call Ollama
        response = self._ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        )
        
        raw_output = response["message"]["content"]
        logger.debug(f"LLM raw output: {raw_output}")
        
        # Validate output structure
        try:
            validated = RendererOutput(rendered_text=raw_output)
        except ValueError as e:
            logger.error(f"LLM output validation failed: {e}")
            raise GuardrailError(f"Output validation failed: {e}")
        
        # Validate no hallucination
        if not validate_no_hallucination(validated.rendered_text, input_data):
            raise GuardrailError("Hallucination detected in LLM output")
        
        return validated.rendered_text
    
    def render_safe(self, explanation: Explanation) -> str:
        """
        Render with fallback to template summary on any error.
        
        This is the recommended method for production use - it never fails,
        falling back to the deterministic template if LLM rendering fails.
        """
        if not self.is_available():
            logger.debug("LLM not available, using template summary")
            return explanation.summary
        
        try:
            return self.render(explanation)
        except (GuardrailError, Exception) as e:
            logger.warning(f"LLM rendering failed ({e}), falling back to template")
            return explanation.summary
