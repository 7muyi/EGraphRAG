ENTITY_EXTRACTION_PROMPT="""Given a text document and a list of entity types, identify all entities of those types from the text.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity in text
- entity_type: One of the following types: {entity_types}
- entity_description: Comprehensive description of the entity

2. Output in JSON format without annotations:
[
    {{
        "name": <entity_name>,
        "type": <entity_type>,
        "description": <entity_description>
    }},
    ...
]

-Real Data-
entity_types:
{entity_types}

text:
{input_text}

output:
"""

DESC_EXTRACTION_PROMPT = """Given a text document and a list of entities, extract descriptive information each entity from the text.
-Step-
1. Identify entities in the text and generate entity description.
2. Output in JSON format without annotations:
[
    {{
        "name": <entity_name (from given entity list)>,
        "type": <entity_type (from given entity list)>,
        "description": <entity_description>
    }},
    ...
]

-Real Data-
entities:
{entities}

text:
{input_text}

Output:
"""

RELATION_EXTRACTION_PROMPT = """Given a text document and a list of entities, identify all relations among entities.

-Steps-
1. Identify all pairs of (source_entity, target_entity) that are **related** to each other.
For each pair of related entities, extract the following information:
- source: entity in the given entity list
- target: entity in the given entity list
- relation_description: description of the relation between source entity and target entity
**Note**: Both the source entity and the target entity must come from the given entity list

2. Output in JSON format without annotations:
[
    {{
        "source": <source_entity (from given entity list)>,
        "target": <target_entity (from given entity list)>,
        "description": <relation_description>,
    }},
    ...
]

-Real Data-
entities:
{entities}

text:
{input_text}

output:
"""

CONTINUE_PROMPT = """MANY {target} were missed in the last extraction. Please add the missing entities using the same format.
Output:
"""

LOOP_PROMPT = "Answer YES | NO if there are still {target} that need to be added."