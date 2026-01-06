from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.domain.psl_schema import PSLGloss
from app.core.logger import logger


def glosses_to_chunks(glosses: List[PSLGloss]) -> List[str]:
    documents = []

    for gloss in glosses:
        doc = (
            f"PSL Gloss: {gloss.gloss}\n"
            f"Possible Meanings: {', '.join(gloss.meanings)}\n"
            f"Context Notes: {gloss.context_notes}\n"
            f"Usage Examples: {' | '.join(gloss.examples)}"
        )
        documents.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    chunks = splitter.split_text("\n\n".join(documents))
    logger.info(f"Generated {len(chunks)} semantic PSL chunks")

    return chunks
