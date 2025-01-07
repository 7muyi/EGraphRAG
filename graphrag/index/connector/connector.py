import re
import uuid
from abc import ABC, abstractmethod
from typing import Any

import networkx as nx

from graphrag.index.utils.nlp import SpacyModel
from graphrag.llm import LLM
from graphrag.model.text_unit import TextUnit
from graphrag.prompts.index.sentence_evaluation import \
    SENTENCE_EVALUATION_PROMPT
from graphrag.utils.transform import str2json


class Connector(ABC):
    @abstractmethod
    def connect(self, graph: nx.Graph, text: str) -> list[TextUnit]:
        pass


class SentenceConnector(Connector):
    def __init__(self,
                 llm: LLM,
                 model_name: str = "en_core_web_sm",
                 ignore_case: bool = False,
                 filter_threshold: int = -1) -> None:
        self._nlp = SpacyModel.get_model(model_name)
        self._llm = llm
        self._ignore_case = ignore_case
        self._filter_threshold = filter_threshold
    
    def _split_sent(self, text: str) -> list[str]:
        doc = self._nlp(text)
        return [sent.text for sent in doc.sents]
    
    def _get_pattern(self, word: str) -> str:
        return r"\s*\b" + re.escape(word) + r"\b\s*[!.,?;:']*"
    
    def _sent_evaluate(self, sents: list[str]) -> list[int]:
        self._llm.reset()
        response = self._llm.generate(
            SENTENCE_EVALUATION_PROMPT.format(
                input_text="\n".join(f"{i}. {sent}" for i, sent in enumerate(sents))
            )
        )
        try:
            scores = str2json(response)
            assert len(scores) == len(sents)
        except:
            return []
        return scores
    
    def _connect(self, entities: list[Any], sents: list[str]) -> list[TextUnit]:
        # Evaluate sentences based on information content
        if self._filter_threshold != -1:
            scores = self._sent_evaluate(sents)
        
        text_units = []
        for entity, data in entities:
            # Add `text_units` to each entity node
            if "text_units" not in data:
                data["text_units"] = []
            corpus = []
            for i, sent in enumerate(sents):
                # Filter out sentences with score below the threshold
                if self._filter_threshold != -1 and scores and scores[i] < self._filter_threshold:
                    continue
                # Check whether the entity appears in the sentence
                if self._ignore_case:
                    find = re.search(self._get_pattern(entity.lower()), sent.lower())
                else:
                    find = re.search(self._get_pattern(entity), sent)
                
                if find:
                    # Map sentnce i to entity node
                    corpus.append(i)
            if not corpus:
                # Without ignoring capitalization, it is possible to encounter sentences where entities cannot be found
                continue
            pre = corpus[0]
            content = [sents[pre]]
            for id in corpus[1:]:
                # Check if it is a continuous sentence
                if id == pre + 1:
                    # Splicing consecutive sentences together
                    content.append(sents[id])
                else:
                    # Map continuous sentence to entity node as corpus attribute
                    text_units.append(TextUnit(
                        id=str(uuid.uuid1()),
                        content=" ".join(content)
                    ))
                    content = [sents[id]]
                    data["text_units"].append(text_units[-1].id)
                pre = id
            if content:
                text_units.append(TextUnit(
                    id=str(uuid.uuid1()),
                    content=" ".join(content)
                ))
                data["text_units"].append(text_units[-1].id)
        return text_units
    
    def connect(self, graph, text):
        # Split text into sentences
        sents = self._split_sent(text)
        # Map sentences (continuous) to entity node
        text_units = self._connect(graph.nodes(data=True), sents)
        return text_units