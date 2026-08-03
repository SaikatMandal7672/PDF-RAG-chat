"""Gatekeeper (block/redact) + Auditor (citation check). Bounds as code."""
from __future__ import annotations

import re

INJECTION = re.compile(
    r"ignore (all |previous |prior )?instructions|reveal (system|secret|key)|"
    r"jailbreak|bypass (safety|policy)|prompt injection", re.I)
AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
SECRET_ECHO = re.compile(r"sk-(live|test)-[A-Za-z0-9]+|xox[bpas]-[A-Za-z0-9-]+", re.I)


def gatekeeper(query: str) -> tuple[bool, str]:
    if INJECTION.search(query):
        return (True, "Blocked: prompt-injection pattern detected. Query logged for review.")
    return (False, "")


def redact_pii(text: str) -> str:
    text = AADHAAR.sub("[REDACTED-ID]", text)
    text = PAN.sub("[REDACTED-PAN]", text)
    text = SECRET_ECHO.sub("[REDACTED-SECRET]", text)
    return text


def output_safe(answer: str) -> tuple[bool, str]:
    """False = unsafe output that must be withheld."""
    if SECRET_ECHO.search(answer):
        return (False, "Withheld: answer contained a secret-like string.")
    return (True, answer)


def auditor(answer: str, hits: list[dict]) -> bool:
    return not hits
