SENTENCE_EVALUATION_PROMPT = """Your task is to score each text in the given list based on its informational content, with texts containing more detailed information receiving higher scores. The score range is from 0 to 10, inclusive, and must be an integer.
**Note**: Ensure that each sentence has a score
Return in JSON format only, as follows:
[<sentence_1 score>, <sentence_2 score>, ..., <sentence_n score>]

Text:
{input_text}

Output:
"""