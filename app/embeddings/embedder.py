from langchain.embeddings import HuggingFaceEmbeddings
from app.core.logger import logger


def get_embedding_model():
    logger.info("Loading local embedding model (sentence-transformers)")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
