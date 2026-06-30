from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    doc_id: str
    text: str
    section_id: str
    section_title: str
    chunk_index: int
    keywords: List[str]
    raw_text: str


@dataclass
class Document:
    doc_id: str
    text: str
    doc_type: str
