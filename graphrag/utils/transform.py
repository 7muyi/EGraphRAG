import json
import re


def str2json(text: str):
    # Delete markdown symbol.
    start = 0
    while text[start] not in ("{", "[") and start < len(text): start += 1
    
    end = len(text) - 1
    while text[end] not in ("}", "]") and end >= 0: end -= 1
    
    if start > end:
        return ""
    else:
        text = text[start:end+1]
    # Delete all comments in JSON
    text = re.sub(r"//.*?\n", "\n", text)
    return json.loads(text)