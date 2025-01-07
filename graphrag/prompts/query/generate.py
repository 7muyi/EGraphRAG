KG_JUDGE_PROMPT = """Please determine whether the given knowledge graph contains all the necessary information to fully answer the question. If it does, output "YES"; otherwise, output "NO".
Criteria: The knowledge graph must contain all the information required to answer the question completely.

Knowledge Graph:
{knowledge_graph}

Question:
{question}

Output:
"""

ADDITIONAL_INFO_PROMPT = """What specific information about which entities is required to answer the question? Output in the following JSON format:
{
    "<entity_name (from knowledge graph)>": [<specific information>, ...],
    "<entity_name (from knowledge graph)>": [<specific information>, ...],
    ...
}

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

Knowledge Graph:
{knowledge_graph}

Question:
{question}

Output:
"""