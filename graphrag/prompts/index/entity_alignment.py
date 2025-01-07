ENTITY_ALIGNMENT_PROMPT = """You are given a list of entities, each with a name and description. Your task is to group same entities that refer to the same real-world entity.
**Note**: Focus on key terms in descriptions to determine if entities refer to the same real-world entity.

Return in JSON format only, as follows:
[[<entity_name>, ...], [<entity_name>, ...]]

Entity List:
{entities}

Output:
"""