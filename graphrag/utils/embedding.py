import os
from urllib import response

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str | list[str], model_name: str = "text-embedding-3-small") -> list[list[float]]:
    response =  client.embeddings.create(input=text, model=model_name)
    return [data.embedding for data in response.data]