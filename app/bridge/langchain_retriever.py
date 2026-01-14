from typing import List, Any, Optional
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from app.retrieval.retriever import retrieve_relevant_chunks
from app.domain.diagnostics import RetrievalDiagnostics

class PSLRetriever(BaseRetriever):
    """
    Wraps the deterministic PSL retrieval logic into a LangChain interface.
    Preserves metrics and diagnostics by attaching them to Document metadata.
    """
    vectorstore: Any
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Execute the custom retrieval pipeline and map results to LC Documents.
        Diagnostics are broadcast to every document's metadata to preserve context.
        """
        # Call the existing robust retriever
        results, diagnostics = retrieve_relevant_chunks(query, self.vectorstore, k=3)
        
        lc_docs = []
        for res in results:
            # We embed the session-level diagnostics into each document's metadata
            # This allows downstream components to access 'score_delta' etc.
            metadata = {
                "score": res.score,
                "confidence": res.confidence.value,
                # Broadcast diagnostics
                "top_score": diagnostics.top_score,
                "score_delta": diagnostics.score_delta,
                "result_density": diagnostics.result_density,
                "gloss_diversity": diagnostics.gloss_diversity,
                "failure_reason": diagnostics.failure_reason.value if diagnostics.failure_reason else "NONE"
            }
            
            lc_docs.append(Document(
                page_content=res.content,
                metadata=metadata
            ))
            
        return lc_docs
