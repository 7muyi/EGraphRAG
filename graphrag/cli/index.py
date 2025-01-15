from graphrag.index import Builder
from graphrag.index.aligner import *
from graphrag.index.connector import SentenceConnector
from graphrag.index.extractor import *
from graphrag.index.text_splitter import TokenTextSplitter
from graphrag.llm import OpenAIModel
from graphrag.utils.config import get_config


def get_builder(config_path: str):
    config = get_config(config_path)
    
    # Load LLM
    if "gpt" in config["llm"]:
        llm = OpenAIModel(config["llm"])
    else:
        # TODO: Load other LLM
        pass
    
    entity_extractors = []
    for extractor in config["entity_extractors"]:
        if extractor["extractor"] == "llm":
            entity_extractors.append(LLMEntityExtractor(
                llm,
                config["entity_types"],
                **extractor.get("params", {})
            ))
        elif extractor["extractor"] == "ner":
            entity_extractors.append(NEREntityExtractor(
                llm,
                extractor["params"]["ner_model"],
                config["entity_types"],
            ))
    
    relation_extractors = []
    for extractor in config["relation_extractors"]:
        if extractor["extractor"] == "llm":
            relation_extractors.append(LLMRelationExtractor(
                llm,
                **extractor.get("params", {})
            ))
    extractor = GraphExtractor(
        ent_extractors=entity_extractors,
        rel_extractors=relation_extractors,
    )
    
    if config["splitter"]["name"] == "token":
        splitter = TokenTextSplitter(**config["splitter"]["params"])
    
    align_pipeline = None
    if "align" in config:
        pipeline_config = config["align"]
        for method in config["align"]:
            if method["method"] == "llm":
                method["params"] = {"llm": llm}
        align_pipeline = AlignPipeline.from_dict(pipeline_config)
    
    connector = None
    if "connector" in config:
        connector_config = config["connector"]
        connector_config["llm"] = llm
        connector = SentenceConnector(**connector_config)
    
    return Builder(
        llm,
        splitter,
        extractor,
        connector,
        align_pipeline,
    )