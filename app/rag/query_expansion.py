import re

from app.rag.utils import clean_text


def expand_query(
    query,
    llm_service
):
    prompt = f"""
Generate exactly two alternative search queries
for the following question.

Do not answer the question.

Only rewrite the question into two useful
search queries.

Question:
{query}

Return exactly two lines.
"""

    response = llm_service.generate([
        {
            "role": "user",
            "content": prompt
        }
    ])

    lines = [
        clean_text(line)
        for line in response.splitlines()
        if clean_text(line)
    ]

    queries = []

    for line in lines:

        line = re.sub(
            r"^\s*(?:[-*]|\d+[.)])\s*",
            "",
            line
        )

        line = clean_text(line)

        if line:
            queries.append(line)

    queries = queries[:2]

    while len(queries) < 2:
        queries.append(query)

    return queries