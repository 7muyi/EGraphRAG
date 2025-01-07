from abc import ABC, abstractmethod


class LLM(ABC):
    def __init__(self):
        self.messages = []

    @abstractmethod
    def generate(self, input: str) -> str:
        pass

    def reset(self):
        self.messages = []