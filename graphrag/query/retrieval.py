import networkx as nx
import numpy as np

from graphrag.llm import LLM
from graphrag.model import Entity, TextUnit
from graphrag.prompts.query.entity_extraction import ENTITY_EXTRACTION
from graphrag.utils.embedding import get_embedding
from graphrag.utils.retrieval import get_cos_sim_matrix, retrieve
from graphrag.utils.transform import str2json


def extract_entities(text: str, llm: LLM) -> list[str]:
    # Extract entity from input text
    llm.reset()
    response = llm.generate(ENTITY_EXTRACTION.format(input_text=text))
    return str2json(response)

def retrieve_entities(query: str, cand_entities: list[str], all_entities: list[Entity], threshold: float = 0.75) -> list[Entity]:
    """Retrieve relevant entities from the entity set through queries and candidate entities"""
    # Retrieve entities based on candidate entities.
    name2ent = {entity.name.lower(): entity for entity in all_entities}
    retrieved_entities = set()
    remains= []
    for entity in cand_entities:
        # ?: Should case sensitivity be ignored?
        if entity.lower() in name2ent:
            retrieved_entities.append(entity.lower())
        else:
            remains.append(entity)
    if remains:
        # Retrieve most similar three entities with similarity above the threshold from entity set.
        indices, _ = retrieve(
            query=np.array(get_embedding(remains)),
            target=np.array([entity.embedding for entity in all_entities]),
            top_k=3,
            threshold=threshold,
        )
        retrieved_entities.update(all_entities[id].name for ids in indices for id in ids)
    
    # Retrieve entities based on query.
    query_embedding = np.array(get_embedding(query))
    entities_embedding = np.array([entity.embedding for entity in all_entities])
    # Retrieve entities with similarity above the threshold from entity set. 
    indices, _ = retrieve(query_embedding, entities_embedding, threshold=threshold)
    retrieved_entities.update(all_entities[id].name for id in indices[0])
    return [name2ent[name] for name in retrieved_entities]

def retrieve_subgraph(query: str, graph: nx.Graph, nodes: list[str], threshold: float) -> nx.Graph:
    query_embedding = np.array(get_embedding(query))
    cur_idx = 0
    while cur_idx < len(nodes):
        u = nodes[cur_idx]
        # Traverse neighboring nodes
        for v in graph.neighbors(u):
            # Not visited before
            if v not in nodes:
                # Add neighbor node v which possess relation with node u, relating to query
                embeddings = np.array([graph.edges[u, v]["attr"].embedding])
                if get_cos_sim_matrix(query_embedding, embeddings)[0][0] > threshold:
                    nodes.append(v)
        # Add similar entity node
        if graph.nodes[u]["attr"].alias:
            for alias in graph.nodes[u]["attr"].alias:
                if alias not in nodes:
                    nodes.append(alias)
        cur_idx += 1
    return graph.subgraph(nodes)

def retrieve_text_units(queries: list[str], text_units: list[TextUnit], **kwds) -> list[list[str]]:
    indices, _ = retrieve(
        np.array(get_embedding(queries)),
        np.array([text_unit.embedding for text_unit in text_units]),
        **kwds
    )
    return [[text_units[id].content for id in ids] for ids in indices]