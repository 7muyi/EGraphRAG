from abc import ABC, abstractmethod
from typing import List

import spacy
import tiktoken


class TextSplitter(ABC):
    @abstractmethod
    def split_text(self, text: str):
        pass


class SentenceSplitter(TextSplitter):
    def __init__(self, chunk_size: int, encoding_name: str, model_name: str) -> None:
        self._tokenizer = tiktoken.get_encoding(encoding_name)
        self.chunk_size = chunk_size
        
        # check if the model_name is valid
        assert (model_name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"),
                f"model {model_name} not supported.")
        try:
            self._nlp = spacy.load(model_name, enable="senter")
        except OSError:
            # download model
            from spacy.cli import download
            download(model_name)
            self._nlp = spacy.load(model_name, enable="senter")
    
    def sent_split(self, text: str) -> List[str]:
        doc = self._nlp(text)
        return list(doc.sents)
    
    def split_text(self, text: str):
        if text == "":
            return []
        
        sents = self.sent_split(text)
        chunks = []
        cur_chunk = []
        cur_num_token = 0
        
        for sent in sents:
            sent_num_token = self.num_token(sent)
            # 0/1 control whether to include sentence delimiter " "
            cur_num_token += sent_num_token + (1 if cur_chunk else 0)
            
            if cur_num_token <= self.chunk_size:
                cur_chunk.append(sent)
            else:
                chunks.append(cur_chunk)
                cur_chunk = [sent]
                cur_num_token = sent_num_token
        if cur_chunk:
            chunks.append(cur_chunk)
        return chunks
    
    def num_token(self,text: str) -> int:
        return len(self._tokenizer.encode(text))


class TokenTextSplitter(TextSplitter):
    def __init__(self, chunk_size: int, over_lap: int, encoding_name: str) -> None:
        self._tokenizer = tiktoken.get_encoding(encoding_name)
        self._chunk_size = chunk_size
        self._over_lap = over_lap
    
    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        return len(self._tokenizer.encode(text))
    
    def split_text(self, text: str):
        splits = []
        input_ids = self._tokenizer.encode(text=text)
        for i in range(0, len(input_ids), self._chunk_size - self._over_lap):
            chunk_ids = input_ids[i:i+self._chunk_size]
            splits.append(self._tokenizer.decode(chunk_ids))
        return splits