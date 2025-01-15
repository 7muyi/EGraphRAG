KG_JUDGE_PROMPT = """Please determine whether the given knowledge graph contains all the necessary information to fully answer the question. If it does, output "YES"; otherwise, output "NO".
Criteria: The knowledge graph must contain all the information required to answer the question completely.

Question:
{question}

Knowledge Graph:
{knowledge_graph}

Output:
"""

ADDITIONAL_INFO_PROMPT = """Given a knowledge graph and a specific question, identify which additional information about which entities is required to fully answer the question, beyond the entity-relationship information already available in the knowledge graph.
**Note**:
1. The output entity must be in the entity list of knowledge graph
2. Output in JSON format, no other annotations are required:
{{
    "<entity_name (from knowledge graph's entity list)>": [<required information>, ...],
    "<entity_name (from knowledge graph's entity list)>": [<required information>, ...],
    ...
}}

##### Example #####
Input:
Question:
How old was Kaiming He when he published ResNet?

Knowledge Graph:
{{
    "entities": ["Kaiming He", "ResNet"],
    "relations": [
        {{
            "source": "Kaiming He",
            "target": "ResNet",
            "description": "He Kaiming is one of the main authors of ResNet."
        }}
    ]
}}

Output:
{{
    "Kaiming He": ["Kaiming He's date of birth"],
    "ResNet": ["The publication date of ResNet"]
}}

##### Real Data #####
Question:
{question}

Knowledge Graph:
{knowledge_graph}

Output:
"""

ATTRIBUTE_EXTRACT = """Extract the entity information based on the provided context. If the required information is not present in the context, output 'NO'; otherwise, output the extracted required information.
**Note**: You must either output 'NO' or the extracted information. Do not output anything else.
Entity: {entity}
Information: {attribute}

Context:
{context}

Output:
"""

EKG_ANSWER_PROMPT = """Answer the question based on the given knowledge graph. If you can not answer just output "I CAN NOT ANSWER."

Question:
{question}

Knowledge Graph:
{knowledge_graph}

Output:
"""

TEXT_ANSWER_PROMPT = """Answer the question based on the given contexxt. If you can not answer just output "I CAN NOT ANSWER."

Question:
{question}

Context:
{context}

Output:
"""