from graphrag.llm import OpenAIModel
from graphrag.query.query import generate
from graphrag.utils.config import get_config


def query(query: str, data_dir: str, config_path: str) -> tuple[str, str]:
    config = get_config(config_path)
    if "gpt" in config["llm"]:
        llm = OpenAIModel(config["llm"])
    else:
        pass
    return generate(query, llm, data_dir, config["threshold"])