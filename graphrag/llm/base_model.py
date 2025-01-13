from abc import ABC, abstractmethod


class LLM(ABC):
    def __init__(self):
        # TODO: Optimize the implementation of multiple rounds of dialogue
        self.messages = []

    @abstractmethod
    def generate(self, input: str) -> str:
        pass

    def reset(self):
        self.messages = []