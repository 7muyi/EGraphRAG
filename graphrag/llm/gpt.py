import logging
import os
import time

from openai import OpenAI

from .base_model import LLM


class OpenAIModel(LLM):
    def __init__(self,
                 model: str = "gpt-4o-mini",
                 max_trials: int = 5,
                 failure_sleep_time: int = 3):
        super().__init__()
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.max_trials = max_trials
        self.failure_sleep_time = failure_sleep_time
        self._sys_prompt = {"role": "system", "content": "You are a helpful assistant."}
        self.reset()
    
    def _generate(self, input: str, messages: list[dict[str, str]] = None) -> str:
        for _ in range(self.max_trials):
            try:
                messages.append({"role": "user", "content": input})
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                )
                response = response.choices[0].message.content
                return response
            except Exception as e:
                logging.error("OpenAI API call failed due to %s. Retrying %d / %d times...", e, _+1, self.max_trials)
                time.sleep(self.failure_sleep_time)
        return ""
    
    def single_turn(self, input: str) -> str:
        return self._generate(input, messages=[self._sys_prompt])
    
    def multi_turn(self, input: str) -> str:
        response = self._generate(input, messages=self.messages)
        self.messages.append({"role": "assistant", "content": response})
        return response
    
    def reset(self):
        self.messages = [self._sys_prompt]