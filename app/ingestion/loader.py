import json
from pathlib import Path
from typing import List
from app.domain.psl_schema import PSLGloss
from app.core.logger import logger


def load_psl_glosses(path: Path) -> List[PSLGloss]:
    logger.info(f"Loading PSL gloss data from {path}")
    raw_data = json.loads(path.read_text(encoding="utf-8"))

    glosses = [PSLGloss(**item) for item in raw_data]
    logger.info(f"Loaded {len(glosses)} PSL gloss entries")

    return glosses
