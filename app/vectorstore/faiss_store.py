from langchain_community.vectorstores import FAISS
from typing import List
from app.core.logger import logger


def build_faiss_index(text_chunks: List[str], embedding_model):
    logger.info("Building FAISS vector store from PSL chunks")

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embedding_model
    )

    logger.info("FAISS index built successfully")
    return vectorstore
