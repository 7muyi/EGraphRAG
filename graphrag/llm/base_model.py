from abc import ABC, abstractmethod


class LLM(ABC):
    def __init__(self):
        self.messages = []
    
    @abstractmethod
    def _generate(self, input: str, messages: list[dict[str, str]] = None) -> str:
        pass
    
    @abstractmethod
    def single_turn(self, input: str) -> str:
        pass
    
    @abstractmethod
    def multi_turn(self, input: str) -> str:
        pass
    
    @abstractmethod
    def reset(self):
        pass