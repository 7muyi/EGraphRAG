from typing import Any

import networkx as nx

from .strategy import EntityExtractor, RealtionExtractor


class GraphExtractor:
    def __init__(self, ent_extractors: list[EntityExtractor], rel_extractors: list[RealtionExtractor]):
        self._ent_extractors = ent_extractors
        self._rel_extractors = rel_extractors
    
    def _process_results(self, entities: list[Any], relations: list[Any]) -> nx.Graph:
        graph = nx.Graph()
        # Add node
        for entity in entities:
            # Nodes with the same name and type are regarded as identical nodes
            if entity["name"] in graph.nodes() and entity["type"] == graph.nodes[entity["name"]]["type"]:
                graph.nodes[entity["name"]]["description"].append(entity["description"])
            else:
                graph.add_node(
                    entity["name"],
                    type=entity["type"],
                    description=[entity["description"]]
                )
        
        # Add edge
        for relation in relations:
            source = relation["source"]
            target = relation["target"]
            if source not in graph.nodes():
                graph.add_node(
                    source,
                    type="",
                    description=[],
                )
            if target not in graph.nodes():
                graph.add_node(
                    target,
                    type="",
                    description=[],
                )
            if graph.has_edge(source, target):
                graph.edges[source, target]["relations"].append(relation["description"])
            else:
                graph.add_edge(
                    source, 
                    target,
                    relations=[relation["description"]]
                )
        return graph
    
    def run(self, text: str) -> nx.Graph:
        entities = []
        relations = []
        
        # Extract entities
        for extractor in self._ent_extractors:
            entities.extend(extractor(text))
        
        # Extract relations
        for extractor in self._rel_extractors:
            relations.extend(extractor(text, entities))
        
        return self._process_results(entities, relations)