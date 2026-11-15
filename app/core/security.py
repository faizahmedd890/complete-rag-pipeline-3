import re


def security_check(query):
    query_lower = query.lower().strip()

    security_patterns = {

        "Sensitive Data": [
            r"\bsalary\b",
            r"\bsalaries\b",
            r"\bpayroll\b",
            r"\bcompensation\b",
            r"\bannual pay\b",
            r"\bhow much does .* earn\b",
            r"\bhow much .* make\b"
        ],

        "PII Leakage": [
            r"\bemail\b",
            r"\be-mail\b",
            r"\bphone\b",
            r"\bphone number\b",
            r"\bmobile number\b",
            r"\bcontact number\b",
            r"\bpersonal details\b",
            r"\bprivate details\b",
            r"\bpersonal information\b",
            r"\bprivate information\b"
        ],

        "Prompt Injection": [
            r"ignore previous instructions",
            r"ignore all previous instructions",
            r"disregard previous instructions",
            r"disregard all previous instructions",
            r"forget previous instructions",
            r"forget all instructions",
            r"override instructions",
            r"bypass security",
            r"system update",
            r"new system prompt",
            r"override privacy",
            r"reveal system prompt",
            r"show system prompt"
        ],

        "Jailbreak": [
            r"developer mode",
            r"pretend you are",
            r"act as",
            r"role-play as",
            r"unrestricted",
            r"no restrictions",
            r"bypass all restrictions",
            r"jailbreak",
            r"do anything now",
            r"\bdan mode\b"
        ],

        "Unauthorized Access": [
            r"not authorized",
            r"no permission",
            r"without permission",
            r"without authorization",
            r"ordinary employee",
            r"normal employee",
            r"confidential payroll",
            r"private employee data",
            r"employee private data",
            r"employee confidential data"
        ]
    }

    for category, patterns in (
        security_patterns.items()
    ):

        for pattern in patterns:

            if re.search(
                pattern,
                query_lower
            ):
                return {
                    "allowed": False,
                    "category": category,
                    "reason": pattern
                }

    return {
        "allowed": True,
        "category": "Allowed",
        "reason": ""
    }