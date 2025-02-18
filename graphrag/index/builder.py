import os
from typing import Any

import networkx as nx
import pandas as pd

from graphrag.llm import LLM
from graphrag.model import *
from graphrag.prompts.index.pronoun_replacement import PRONOUN_REPLACE_PROMPT
from graphrag.prompts.index.summary import (
    ENTITY_DESCRIPTION_SUMMARY_PROMPT, RELATION_DESCRIPTION_SUMMARY_PROMPT)
from graphrag.utils.embedding import get_embedding

from .aligner import AlignPipeline
from .connector import Connector
from .extractor import GraphExtractor
from .text_splitter import TextSplitter


class Builder:
    def __init__(self,
                 llm: LLM,
                 text_splitter: TextSplitter,
                 extractor: GraphExtractor,
                 connector: Connector = None,
                 align_pipeline: AlignPipeline = None):
        self._llm = llm
        self._text_splitter = text_splitter
        self._extractor = extractor
        self._connector = connector
        self._align_pipeline = align_pipeline
    
    def _pronoun_replace(self, text: str, max_length: int = 5000) -> str:
        res = []
        # To avoid text exceeding the input length limit, replace it in blocks
        for i in range(0, len(text), max_length):
            response = self._llm.single_turn(PRONOUN_REPLACE_PROMPT.format(input_text=text[i:i+max_length]))
            res.append(response)
        return "".join(res)
    
    def _load_doc(self, doc_or_path: str, replace_pronoun: bool = False) -> str:
        if os.path.exists(doc_or_path):
            if replace_pronoun:
                # Check if there are processed doc files under this folder
                doc_file_name = os.path.basename(doc_or_path).split(".")[0]
                doc_dir = os.path.dirname(doc_or_path)
                processed_doc_path = os.path.join(doc_dir, f"{doc_file_name}_processed.txt")
                if os.path.exists(processed_doc_path):
                    # If it exist, read the processed file directly
                    text = open(processed_doc_path, "r", encoding="utf-8").read()
                else:
                    # Otherwise process original file and save the processed data
                    raw_content = open(doc_or_path, "r", encoding="utf-8").read()
                    text = self._pronoun_replace(raw_content)
                    with open(processed_doc_path, "w", encoding="utf-8", newline="") as f:
                        f.write(text)
            else:
                text = open(doc_or_path, "r", encoding="utf-8").read()
        else:
            text = self._pronoun_replace(doc_or_path) if replace_pronoun else doc_or_path
        return text.replace("\n\n", "\n")
    
    def _merge(self, targ: nx.Graph, subgraph: nx.Graph):
        # Merge node
        for node, data in subgraph.nodes(data=True):
            # Existing node, smae name and type
            if node in targ.nodes() and data["type"] == targ.nodes[node]["type"]:
                # Merge `description` and `corpus`
                targ.nodes[node]["description"].extend(data["description"])
                if self._connector:
                    targ.nodes[node]["text_units"].extend(data["text_units"])
            else:
                targ.add_node(node, **data)
        # Merge edge
        for source, target, data in subgraph.edges(data=True):
            if targ.has_edge(source,target):
                targ.edges[source, target]["relations"].extend(data["relations"])
            else:
                targ.add_edge(source, target, **data)
    
    def _ent_summary(self, entity: str, descs: list[str]) -> str:
        if not descs:
            return "None"
        if len(descs) == 1:
            return descs[0]
        response = self._llm.single_turn(
            ENTITY_DESCRIPTION_SUMMARY_PROMPT.format(
                entity=entity,
                descriptions="\n".join(f"{i + 1}: {desc}" for i, desc in enumerate(descs))
        ))
        return response
    
    def _rel_summary(self, source: str, target: str, descs: list[str]) -> str:
        if not descs:
            return "None"
        if len(descs) == 1:
            return descs[0]
        response = self._llm.single_turn(
            RELATION_DESCRIPTION_SUMMARY_PROMPT.format(
                source=source,
                target=target,
                descriptions="\n".join(f"{i + 1}: {desc}" for i, desc in enumerate(descs))
        ))
        return response
    
    def _save(self, data: list[Any], output: str):
        df = pd.DataFrame([record.__dict__ for record in data])
        df.to_parquet(output, engine="pyarrow")
    
    def run(self, doc_or_path: str, output: str, replace_pronoun: bool = False):
        text_units = []
        graph = nx.Graph()
        # Step 1: Load document
        text = self._load_doc(doc_or_path, replace_pronoun)
        # Step 2: Split document into chunks
        chunks = self._text_splitter.split_text(text)
        
        for chunk in chunks:
            # Step 3: Extract KG from each chunk
            g = self._extractor.run(chunk)
            # Step 4: Connect chunk to KG
            if self._connector:
                text_units.extend(self._connector.connect(g, chunk))
            # Step 5: Merge `g` to `graph`
            self._merge(graph, g)
        
        entities: list[Entity] = []
        relations: list[Relation] = []
        
        # Summary description for each entity
        description_list = [
            self._ent_summary(node, data["description"])
            for node, data in graph.nodes(data=True)
        ]
        # Embedding, batch processing
        embeddings = get_embedding(description_list)
        for i, (node, data) in enumerate(graph.nodes(data=True)):
            entities.append(Entity(
                id=str(len(entities)),
                name=node,
                type=data["type"],
                description=description_list[i],
                embedding=embeddings[i],
                text_units=data.get("text_units"),
            ))
        
        # Summary description for each relation
        description_list = [
            self._rel_summary(source, target, data["relations"])
            for source, target, data in graph.edges(data=True)
        ]
        # Embedding, batch processing
        embeddings = get_embedding(description_list)
        ent2id = {entity.name: entity.id for entity in entities}
        for i, (source, target, data) in enumerate(graph.edges(data=True)):
            relations.append(Relation(
                id=str(len(relations)),
                source=ent2id[source],
                target=ent2id[target],
                description=description_list[i],
                embedding=embeddings[i]
            ))
        
        # Step 6: Align entities
        if self._align_pipeline:
            self._align_pipeline.run(entities)
        # Step 7: Save result
        if not os.path.exists(output):
            os.makedirs(output)
        self._save(entities, os.path.join(output, "entities.parquet"))
        self._save(relations, os.path.join(output, "relations.parquet"))
        self._save(text_units, os.path.join(output, "text_units.parquet"))
