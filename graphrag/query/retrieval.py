from typing import List

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

def retrieve_entities(entities: list[str], all_entities: list[Entity]) -> list[Entity]:
    name2ent = {entity.name.lower(): entity for entity in all_entities}
    retrieved_entities = []
    remains= []
    for entity in entities:
        # ?: Should case sensitivity be ignored?
        if entity.lower() in name2ent:
            retrieved_entities.append(name2ent[entity.lower()])
        else:
            # If extracted entity not in entity set
            remains.append(entity)
    if remains:
        # Retrieve most similari three entities from entity set
        _, max_ids = retrieve(
            query=np.array(get_embedding(remains)),
            target=np.array([entity.embedding for entity in all_entities]),
            top_k=3,
        )
        retrieved_entities.extend(all_entities[id] for id in set(id for ids in max_ids for id in ids))
    return retrieved_entities

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

def retrieve_text_units(queries: list[str], text_units: list[TextUnit], threshold: float) -> list[list[str]]:
    sim_matrix = get_cos_sim_matrix(
        x=np.array(get_embedding(queries)),
        y=np.array([text_unit.embedding for text_unit in text_units]),
    )
    return [[text_units[i].content for i, sim in enumerate(q2ts) if sim > threshold] for q2ts in sim_matrix]