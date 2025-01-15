import logging

from graphrag.llm import LLM
from graphrag.prompts.query.generate import (ADDITIONAL_INFO_PROMPT,
                                             ATTRIBUTE_EXTRACT,
                                             EKG_ANSWER_PROMPT,
                                             KG_JUDGE_PROMPT,
                                             TEXT_ANSWER_PROMPT)
from graphrag.utils.transform import str2json

from .loader import load_graph
from .retrieval import *


def generate(query: str, llm: LLM, data_dir: str, threshold: float):
    # Load data and construct graph
    graph, entities, text_units = load_graph(data_dir)
    id2ent = {entity.id: entity for entity in entities}
    name2ent = {entity.name: entity for entity in entities}
    id2text_units = {text_unit.id: text_unit for text_unit in text_units}
    
    # Extract entities from query
    extracted_entities = extract_entities(query, llm)
    logging.info("Extracted entities: %s", ", ".join(extracted_entities))
    # Retrieve entity from entity set
    retrieved_entities = retrieve_entities(query, extracted_entities, entities)
    logging.info("Retrieved entities: %s", ", ".join(entity.name for entity in retrieved_entities))
    nodes = [entity.id for entity in retrieved_entities]
    if not nodes:
        # Become Text RAG
        contexts = retrieve_text_units(
            query,
            text_units,
            top_k=7,
        )[0]
        response = llm.single_turn(TEXT_ANSWER_PROMPT.format(question=query, context="\n".join(contexts)))
        logging.info(f"Context:\n%s", str(contexts))
        logging.info(f"Response:\n%s", response)
        return contexts, response
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
    llm.reset()  # Just in case
    response = llm.multi_turn(
        KG_JUDGE_PROMPT.format(knowledge_graph=str(kg_context), question=query),
    )
    if response == "YES":
        response = llm.multi_turn(EKG_ANSWER_PROMPT.format(knowledge_graph=str(kg_context), question=query))
        logging.info(f"Context:\n%s", str(kg_context))
        logging.info(f"Response:\n%s", response)
        return kg_context, response
    # Extract attributes of entity required
    response = llm.multi_turn(ADDITIONAL_INFO_PROMPT.format(question=query, knowledge_graph=kg_context))
    response = str2json(response)
    extracted_attrs = {}
    for entity, attributes in response.items():
        if entity not in kg_context["entities"] or not attributes or not name2ent[entity].text_units:
            continue
        text_units4ent = [
            id2text_units[text_unit_id]
            for text_unit_id in name2ent[entity].text_units
        ]
        # TODO: improve text retrieval quality.
        contexts = retrieve_text_units(
            [f"{entity}: {attribute}" for attribute in attributes],
            text_units4ent,
            # threshold=threshold
            top_k=3,
        )
        extracted_attrs[entity] = []
        for i, attribute in enumerate(attributes):
            logging.info("%s: %s", entity, attribute)
            logging.info("Context:\n%s", "\n".join(contexts[i]))
            response = llm.multi_turn(
                ATTRIBUTE_EXTRACT.format(entity=entity, attribute=attribute, context="\n".join(contexts[i]))
            )
            if "NO" in response:
                extracted_attrs[entity].append(contexts[i][0])
            else:
                extracted_attrs[entity].append(response)
    kg_context["entities"] = []
    for entity in subgraph.nodes():
        kg_context["entities"].append({
            "name": id2ent[entity].name
        })
        if extracted_attrs.get(id2ent[entity].name):
            kg_context["entities"][-1]["information"] = extracted_attrs.get(id2ent[entity].name)
    contexts = str(kg_context)
    response = llm.multi_turn(EKG_ANSWER_PROMPT.format(knowledge_graph=contexts, question=query))
    logging.info("Context:\n%s", contexts)
    messages = "\n".join([f"{turn['role']}: {turn['content']}" for turn in llm.messages if turn["role"] == "assistant"])
    logging.info("Response:\n%s", messages)
    return contexts, response.replace("\n\n", "\n")