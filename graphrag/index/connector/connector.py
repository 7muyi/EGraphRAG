import hashlib
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any

import networkx as nx
import tiktoken

from graphrag.index.utils.nlp import SpacyModel
from graphrag.llm import LLM
from graphrag.model.text_unit import TextUnit
from graphrag.prompts.index.sentence_evaluation import \
    SENTENCE_EVALUATION_PROMPT
from graphrag.utils.embedding import get_embedding
from graphrag.utils.transform import str2json


class Connector(ABC):
    @abstractmethod
    def connect(self, graph: nx.Graph, text: str) -> list[TextUnit]:
        pass


class SentenceConnector(Connector):
    def __init__(self,
                 llm: LLM,
                 model_name: str = "en_core_web_sm",
                 encoding_model: str = "cl100k_base") -> None:
        self._llm = llm
        self._MAX_LENGTH = 512
        self._nlp = SpacyModel.get_model(model_name)
        self._encoding_model = tiktoken.get_encoding(encoding_model)
    
    def _get_sents(self, text: str) -> list[str]:
        """Split the text into sentences."""
        doc = self._nlp(text)
        return [sent.text for sent in doc.sents]
    
    def _get_pattern(self, word: str) -> str:
        """Generate regular expressions for word matching."""
        return r"\s*\b" + re.escape(word) + r"\b\s*[!.,?;:']*"
    
    def _sent_evaluate(self, sents: list[str]) -> list[int]:
        """Using LLM to evaluate the information content of sentences."""
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
    
    def _num_tokens(self, text: str) -> int:
        """Return the number of tokens in a text string."""
        return len(self._encoding_model.encode(text))
    
    def _get_hash(self, nums: list[int]) -> str:
        """Hash encode a list of numbers"""
        rep = f"{len(nums)}-" + "-".join(map(str, nums))
        return hashlib.sha256(rep.encode()).hexdigest()
    
    def _merge_nums(self, nums: list[str]) -> list[list[str]]:
        """Merge consecutive numbers into a list"""
        pre = nums[0]
        result = [[pre]]
        for num in nums[1:]:
            if num == pre + 1:
                result[-1].append(num)
            else:
                result.append([num])
            pre = num
        return result
    
    def _connect(
        self,
        entities: list[Any],
        sents: list[str],
        ignore_case: bool = True,
        threshold: int = -1) -> list[TextUnit]:
        """Building connections between entities and sentences."""
        # Evaluate sentences based on information content
        if threshold != -1:
            scores = self._sent_evaluate(sents)
        
        text_units = {}
        for entity, data in entities:
            if "text_units" not in data:
                data["text_units"] = []
            corpus = []
            # Search for sentences with entities appearing
            for i, sent in enumerate(sents):
                # Filter out sentences with score below the threshold
                if threshold != -1 and scores and scores[i] < threshold:
                    continue
                # Check whether the entity appears in the sentence
                if ignore_case:
                    find = re.search(self._get_pattern(entity.lower()), sent.lower())
                else:
                    find = re.search(self._get_pattern(entity), sent)
                
                if find:
                    corpus.append(i)
            if not corpus:
                # Without ignoring capitalization, it is possible to encounter sentences where entities cannot be found
                continue
            # Merge consecutive sentences
            ids = self._merge_nums(corpus)
            # Construct TextUnits, each consisting of consecutive sentences with a length not exceeding MAX_LENGTH
            for sents_id in ids:
                final_sents_id = [sents_id[0]]
                length = self._num_tokens(sents[sents_id[0]])
                for sent_id in sents_id[1:]:
                    length_ = length + self._num_tokens(sents[sent_id])
                    if length_ <= self._MAX_LENGTH:
                        final_sents_id.append(sent_id)
                        length = length_
                    else:
                        hash_id = self._get_hash(final_sents_id)
                        # Duplicate removal
                        if hash_id not in text_units:
                            text_units[hash_id] = TextUnit(
                                id=str(uuid.uuid1()),
                                content=" ".join(sents[id] for id in final_sents_id)
                            )
                        data["text_units"].append(text_units[hash_id].id)
                        length = length_ - length
                        final_sents_id = [sent_id]
                # Post processing
                hash_id = self._get_hash(final_sents_id)
                if hash_id not in text_units:
                    text_units[hash_id] = TextUnit(
                        id=str(uuid.uuid1()),
                        content=" ".join(sents[id] for id in final_sents_id)
                    )
                data["text_units"].append(text_units[hash_id].id)
        text_units = [text_unit for _, text_unit in text_units.items()]
        # Perform batch embedding
        embeddings = get_embedding([text_unit.content for text_unit in text_units])
        for i in range(len(text_units)):
            text_units[i].embedding = embeddings[i]
        return text_units
    
    def connect(self, graph, text):
        # Split text into sentences
        sents = self._get_sents(text)
        # Build connection between graph and chunk
        text_units = self._connect(graph.nodes(data=True), sents, ignore_case=False, threshold=5)
        return text_units