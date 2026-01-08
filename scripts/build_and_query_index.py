from pathlib import Path
from app.ingestion.loader import load_psl_glosses
from app.ingestion.chunker import glosses_to_chunks
from app.embeddings.embedder import get_embedding_model
from app.vectorstore.faiss_store import build_faiss_index
from app.retrieval.retriever import retrieve_relevant_chunks
from app.core.logger import logger


def main():
    # Load and preprocess PSL data
    glosses = load_psl_glosses(Path("data/raw/psl_glosses.json"))
    chunks = glosses_to_chunks(glosses)

    # Load embedding model
    embedding_model = get_embedding_model()

    # Build vector store
    vectorstore = build_faiss_index(chunks, embedding_model)

    # Test query
    query = "What does RUN mean in PSL?"
    results = retrieve_relevant_chunks(query, vectorstore)

    logger.info("Top retrieved chunk:")
    logger.info(results[0].page_content)


if __name__ == "__main__":
    main()
