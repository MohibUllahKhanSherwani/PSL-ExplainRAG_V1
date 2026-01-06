from pydantic import BaseModel
from typing import List


class PSLGloss(BaseModel):
    gloss: str
    meanings: List[str]
    context_notes: str
    examples: List[str]
