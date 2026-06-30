from dataclasses import dataclass
from typing import List, Optional, Any


@dataclass
class Chunk:
    doc_id: str
    text: str
    section_id: str
    section_title: str
    chunk_index: int
    keywords: List[str]
    raw_text: str
    section: str
    page: int
    parent_id: Optional[str]
    chunk_id: str
    is_parent: bool
    token_count: int
    rerank_score: Optional[float] = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.__dict__.keys()

    def copy(self) -> "Chunk":
        import copy

        return copy.copy(self)


@dataclass
class Document:
    doc_id: str
    text: str
    doc_type: str
