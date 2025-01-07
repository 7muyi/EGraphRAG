from dataclasses import dataclass
from typing import Any


@dataclass
class TextUnit:
    id: str
    content: str
    embedding: list[float] | None = None

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        content_key: str = "content",
        embedding_key: str = "embedding",
    ) -> "TextUnit":
        return TextUnit(
            id=d[id_key],
            content=d[content_key],
            embedding=d.get(embedding_key),
        )