import logging

from graphrag.llm import LLM
from graphrag.prompts.query.generate import (ADDITIONAL_INFO_PROMPT,
                                             ANSWER_PROMPT, ATTRIBUTE_EXTRACT,
                                             KG_JUDGE_PROMPT)
from graphrag.utils.transform import str2json

from .loader import load_graph
from .retrieval import *


def generate(query: str, llm: LLM, data_dir: str, threshold: float):
    # Load data and construct graph
    logging.info("Loading Enhanced Knowledge Graph...")
    graph, entities, text_units = load_graph(data_dir)
    id2ent = {entity.id: entity for entity in entities}
    name2ent = {entity.name: entity for entity in entities}
    id2text_units = {text_unit.id: text_unit for text_unit in text_units}
    
    # Extract entities from query
    extracted_entities = extract_entities(query, llm)
    logging.info("Extracted entities from query: %s", ", ".join(extracted_entities))
    # Retrieve entity from entity set
    retrieved_entities = retrieve_entities(extracted_entities, entities)
    logging.info("Retrieved entities from knowledge graph: %s", ", ".join(entity.name for entity in retrieved_entities))
    nodes = [entity.id for entity in retrieved_entities]
    # Retrieve subgraph
    subgraph = retrieve_subgraph(query, graph, nodes, threshold)
    # Constrcut context
    kg_context = {"entities": [], "relations": []}
    kg_context = {
        "entities": [id2ent[entity].name for entity in subgraph.nodes()],
        "relations": [
            {
                "source": id2ent[u].name,
                "target": id2ent[v].name,
                "relation": subgraph.edges[u, v]["attr"].description
            }
            for u, v in subgraph.edges()
        ]
    }
    llm.reset()
    response = llm.generate(
        KG_JUDGE_PROMPT.format(knowledge_graph=str(kg_context), question=query),
    )
    if response == "YES":
        return kg_context, llm.generate(ANSWER_PROMPT.format(knowledge_graph=str(subgraph), question=query))
    # Extract attributes of entity required
    response = llm.generate(ADDITIONAL_INFO_PROMPT.format(question=query, knowledge_graph=kg_context))
    response = str2json(response)
    extracted_attrs = {}
    for entity, attributes in response.items():
        if entity not in kg_context["entities"] or not attributes:
            continue
        text_units4ent = [
            id2text_units[text_unit_id]
            for text_unit_id in name2ent[entity].text_units
        ]
        if text_units4ent:
            contexts = retrieve_text_units(
                [f"{entity}: {attribute}" for attribute in attributes],
                text_units4ent,
                threshold
            )
            # extracted_attrs[entity] = contexts
            extracted_attrs[entity] = []
            for i, attribute in enumerate(attributes):
                response = llm.generate(
                    ATTRIBUTE_EXTRACT.format(entity=entity, attribute=attribute, context="\n".join(contexts[i]))
                )
                logging.info("The %s of %s: %s", attribute, entity, response)
                extracted_attrs[entity].append(response)
    kg_context["entities"] = []
    for entity in subgraph.nodes():
        kg_context["entities"].append({
            "name": id2ent[entity].name
        })
        if extracted_attrs.get(id2ent[entity].name):
            kg_context["entities"][-1]["information"] = extracted_attrs.get(id2ent[entity].name)
    contexts = str(kg_context)
    response = llm.generate(ANSWER_PROMPT.format(knowledge_graph=contexts, question=query))
    messages = "\n".join([f"{turn['role']}: {turn['content']}" for turn in llm.messages])
    logging.info("Response:\n%s", messages)
    return contexts, response.replace("\n\n", "\n")