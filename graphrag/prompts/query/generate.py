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
1. The output entity must be in the knowledge graph
2. Output in JSON format, no other annotations are required:
{{
    "<entity_name (from knowledge graph)>": [<required information>, ...],
    "<entity_name (from knowledge graph)>": [<required information>, ...],
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

ATTRIBUTE_EXTRACT = """Extract the information of the entity based on the provided context, **Only** output extracted information.
Entity: {entity}
Information: {attribute}

Context:
{context}

Information:
"""

ANSWER_PROMPT = """Answer the question based on the given knowledge graph. If you can not answer just output "I CAN NOT ANSWER."

Question:
{question}

Knowledge Graph:
{knowledge_graph}

Output:
"""