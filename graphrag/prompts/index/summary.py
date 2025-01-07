ENTITY_DESCRIPTION_SUMMARY_PROMPT = """Given an entity and its descriptive sentences, generate a concise summary that highlights its core features clearly and briefly.
**Note**: Only output the descriptive sentence, no other annotations are required.

Entity:
{entity}

Description:
{descriptions}

Output:
"""

RELATION_DESCRIPTION_SUMMARY_PROMPT = """Given 2 entities and their relation descriptive sentences, generate a concise summary that highlights their relation clearly and briefly.
**Note**: Only output the descriptive sentence, no other annotations are required.

Entity:
source: {source}
target: {target}

Description:
{descriptions}

Output:
"""