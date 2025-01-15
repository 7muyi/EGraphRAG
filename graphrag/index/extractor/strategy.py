from abc import ABC, abstractmethod
from typing import Any

from graphrag.index.utils.nlp import SpacyModel
from graphrag.llm import LLM
from graphrag.prompts.index.extraction import *
from graphrag.utils.transform import str2json


class LLMExtractor:
    def __init__(self, llm: LLM, prompt: str, max_gleanings: int) -> None:
        self._llm = llm
        self._extraction_prompt = prompt
        self._max_gleanings = max_gleanings
    
    def _llm_extract(self, extract_prompt: str, continue_prompt: str, loop_prompt: str) -> list[str]:
        self._llm.reset()  # Just in case
        
        results = []
        response = self._llm.multi_turn(extract_prompt)
        results.append(response)
        
        # ensure to extract as many information as possible
        for _ in range(self._max_gleanings - 1):
            response = self._llm.multi_turn(continue_prompt)
            results.append(response)
            
            # determine there further extraction is necessary
            response = self._llm.multi_turn(loop_prompt)
            if response != "YES":
                break
        return results


class EntityExtractor(ABC):
    def __init__(self, entity_types: list[str] | None = None) -> None:
        self._entity_types = (
            entity_types 
            if entity_types 
            else ["PERSON", "ORGANIZATION", "PRODUCT", "LOCATION", "EVENT"]
        )
    
    @abstractmethod
    def _extract(self, text: str) -> list[Any]:
        pass
    
    def __call__(self, text: str):
        return self._extract(text)


class LLMEntityExtractor(EntityExtractor, LLMExtractor):
    def __init__(self,
                 llm: LLM,
                 entity_types: list[str] | None = None,
                 prompt: str | None = None,
                 max_gleanings: int = 1) -> None:
        EntityExtractor.__init__(self, entity_types)
        LLMExtractor.__init__(self, llm, prompt or ENTITY_EXTRACTION_PROMPT, max_gleanings)
    
    def _extract(self, text: str) -> list[Any]:
        results = self._llm_extract(
            extract_prompt=self._extraction_prompt.format(entity_types=self._entity_types, input_text=text),
            continue_prompt=CONTINUE_PROMPT.format(target="entities"),
            loop_prompt=LOOP_PROMPT.format(target="entities")
        )
        return self._process_results(results)
    
    def _process_results(self, results: list[str]) -> list[Any]:
        """Convert `str` to `json`"""
        entities = []
        for result in results:
            entities.extend(str2json(result))
        return entities


class NEREntityExtractor(EntityExtractor, LLMExtractor):
    def __init__(self, llm: LLM, ner_model: str, entity_types: list[str] | None = None, prompt: str = None) -> None:
        EntityExtractor.__init__(self, entity_types)
        LLMExtractor.__init__(self, llm, prompt or DESC_EXTRACTION_PROMPT, 1)
        self._nlp = SpacyModel.get_model(ner_model)
    
    def _extract(self, text: str) -> list[Any]:
        doc = self._nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in self._entity_types:
                entity = {"name": ent.text, "type": ent.label_}
                # Ignore Duplicate
                if entity not in entities:
                    entities.append(entity)
        return self._extract_desc(text, entities)
    
    def _extract_desc(self, text: str, entities: list[Any]) -> list[Any]:
        # Extract description of entity from text.
        response = self._llm.single_turn(self._extraction_prompt.format(entities=entities, input_text=text))
        return str2json(response)


class RealtionExtractor(ABC):
    @abstractmethod
    def _extract(self, text: str, entities: list[Any]) -> list[Any]:
        pass
    
    def __call__(self, text: str, entities: list[Any]) -> list[Any]:
        return self._extract(text, entities)


class LLMRelationExtractor(RealtionExtractor, LLMExtractor):
    def __init__(self, llm: LLM, prompt: str | None = None, max_gleanings: int = 1) -> None:
        super().__init__(llm, prompt or RELATION_EXTRACTION_PROMPT, max_gleanings)
    
    def _extract(self, text: str, entities: list[Any]) -> list[Any]:
        results = self._llm_extract(
            extract_prompt=self._extraction_prompt.format(
                input_text=text,
                entities=[{"name": entity["name"], "type": entity["type"]} for entity in entities]
            ),
            continue_prompt=CONTINUE_PROMPT.format(target="relations"),
            loop_prompt=LOOP_PROMPT.format(target="relations"),
        )
        return self._process_results(results)
    
    def _process_results(self, results: list[str]) -> list[Any]:
        relations = []
        for result in results:
            relations.extend(str2json(result))
        return relations