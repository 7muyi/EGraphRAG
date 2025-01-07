import os

import networkx as nx

from graphrag.model import *

from .utils.load import load_parquet


def load_graph(data_dir: str):
    # Load data from parquet
    entities = load_parquet(os.path.join(data_dir, "entities.parquet"))
    relations = load_parquet(os.path.join(data_dir, "relations.parquet"))
    text_units = load_parquet(os.path.join(data_dir, "text_units.parquet"))
    
    entities = [Entity.from_dict(entity) for entity in entities]
    relations = [Relation.from_dict(relation) for relation in relations]
    text_units = [TextUnit.from_dict(text_unit) for text_unit in text_units]
    
    # Construct graph
    # Add node
    graph = nx.Graph()
    for entity in entities:
        graph.add_node(entity.id, attr=entity)
    
    # Add edge
    for relation in relations:
        # source = relation.source
        # target = relation.target
        # if graph.has_edge(source, target):
        #     graph.edges[source, target]["relations"].append(relation)
        # else:
        
        graph.add_edge(relation.source, relation.target, attr=relation)
    
    return graph, entities, text_units