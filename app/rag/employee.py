import re

from app.rag.utils import clean_text


def normalize_employee_query(query):
    normalized = query

    patterns = [
        r"\bemployee[\s-]+0*(\d+)\b",
        r"\bemp[\s-]+0*(\d+)\b"
    ]

    employee_numbers = []

    for pattern in patterns:

        for match in re.finditer(
            pattern,
            query,
            re.IGNORECASE
        ):

            try:
                number = int(
                    match.group(1)
                )

                if number > 0:
                    employee_numbers.append(
                        number
                    )

            except ValueError:
                pass

    employee_numbers = sorted(
        set(employee_numbers)
    )

    if not employee_numbers:
        return query, []

    variants = []

    for number in employee_numbers:

        variants.extend([
            f"EMP-{number:03d}",
            f"EMP-{number:02d}",
            f"EMP-{number}",
            f"EMP {number:03d}",
            f"EMP {number:02d}",
            f"EMP {number}",
            f"employee {number}",
            f"employee-{number}"
        ])

    variants = list(
        dict.fromkeys(variants)
    )

    normalized = re.sub(
        r"\bemployee[\s-]+0*\d+\b",
        lambda match:
        f"EMP-{int(re.search(r'\d+', match.group()).group()):03d}",
        normalized,
        flags=re.IGNORECASE
    )

    normalized = re.sub(
        r"\bemp[\s-]+0*\d+\b",
        lambda match:
        f"EMP-{int(re.search(r'\d+', match.group()).group()):03d}",
        normalized,
        flags=re.IGNORECASE
    )

    return normalized, variants


def exact_employee_search(
    query,
    documents
):
    normalized_query, variants = (
        normalize_employee_query(query)
    )

    if not variants:
        return [], normalized_query

    employee_numbers = set()

    for variant in variants:

        match = re.search(
            r"\bEMP[-\s]?0*(\d+)\b",
            variant.upper()
        )

        if match:
            employee_numbers.add(
                int(match.group(1))
            )

    matched_documents = []

    for doc in documents:

        content = clean_text(
            doc.page_content
        )

        if not content:
            continue

        for match in re.finditer(
            r"\bEMP[-\s]?0*(\d+)\b",
            content.upper()
        ):

            try:
                document_number = int(
                    match.group(1)
                )
            except ValueError:
                continue

            if document_number in employee_numbers:
                matched_documents.append(doc)
                break

    unique = []
    seen = set()

    for doc in matched_documents:

        content = clean_text(
            doc.page_content
        )

        if content in seen:
            continue

        seen.add(content)
        unique.append(doc)

    return unique, normalized_query