from app.core.logger import logger


def retrieve_relevant_chunks(query: str, vectorstore, k: int = 3):
    logger.info(f"Retrieving top-{k} chunks for query: {query}")

    results = vectorstore.similarity_search(query, k=k)

    logger.info(f"Retrieved {len(results)} chunks")
    return results
