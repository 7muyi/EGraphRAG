from dataclasses import dataclass
from typing import Any


@dataclass
class Relation:
    """
    Instance of relation between entities.
    
    Attributes:
        id (str): ID of relation.
        source (str): ID of source entity.
        target (str): ID of target entity.
        description (str): the description of relation between source and target.
        embedding (list[float]): the embedding of description.
        reference (str): the sentence support the relation between source and target.
    """
    id: str
    source: str
    target: str
    description: str
    embedding: list[float] | None = None

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        id_key: str = "id",
        source_key: str = "source",
        target_key: str = "target",
        description_key: str = "description",
        embedding_key: str = "embedding",
    ) -> "Relation":
        return Relation(
            id=d[id_key],
            source=d[source_key],
            target=d[target_key],
            description=d[description_key],
            embedding=d.get(embedding_key),
        )