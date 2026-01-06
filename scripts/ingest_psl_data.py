from pathlib import Path
from app.ingestion.loader import load_psl_glosses
from app.ingestion.chunker import glosses_to_chunks
from app.core.logger import logger


def main():
    data_path = Path("data/raw/psl_glosses.json")
    glosses = load_psl_glosses(data_path)
    chunks = glosses_to_chunks(glosses)

    logger.info("Preview of first semantic chunk:")
    logger.info(chunks[0])


if __name__ == "__main__":
    main()
