from __future__ import annotations

import re
from re import Pattern

REDACT_EMAIL = "[REDACTED_EMAIL]"
REDACT_PHONE = "[REDACTED_PHONE]"
REDACT_CARD = "[REDACTED_CARD]"
REDACT_ADDRESS = "[REDACTED_ADDRESS]"
REDACT_ID = "[REDACTED_ID]"
REDACT_IP = "[REDACTED_IP]"
REDACT_URL = "[REDACTED_URL]"
REDACT_ACCOUNT = "[REDACTED_ACCOUNT]"

URL_PATTERN: Pattern[str] = re.compile(r"\bhttps?://[^\s]+", re.IGNORECASE)
EMAIL_PATTERN: Pattern[str] = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IP_PATTERN: Pattern[str] = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)
PHONE_PATTERN: Pattern[str] = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)"
)
CARD_PATTERN: Pattern[str] = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
SSN_PATTERN: Pattern[str] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
ADDRESS_PATTERN: Pattern[str] = re.compile(
    r"\b\d{1,5}\s+(?:[A-Za-z0-9#.-]+\s){0,6}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Boulevard|Blvd|Drive|Dr|Court|Ct|Place|Pl|Way)\b\.?",
    re.IGNORECASE,
)
IBAN_PATTERN: Pattern[str] = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
ACCOUNT_PATTERN: Pattern[str] = re.compile(r"\b\d{9,18}\b")


def redact_text(text: str) -> str:
    redacted = text
    redacted = URL_PATTERN.sub(REDACT_URL, redacted)
    redacted = EMAIL_PATTERN.sub(REDACT_EMAIL, redacted)
    redacted = IP_PATTERN.sub(REDACT_IP, redacted)
    redacted = PHONE_PATTERN.sub(REDACT_PHONE, redacted)
    redacted = SSN_PATTERN.sub(REDACT_ID, redacted)
    redacted = CARD_PATTERN.sub(REDACT_CARD, redacted)
    redacted = IBAN_PATTERN.sub(REDACT_ACCOUNT, redacted)
    redacted = ACCOUNT_PATTERN.sub(REDACT_ACCOUNT, redacted)
    redacted = ADDRESS_PATTERN.sub(REDACT_ADDRESS, redacted)
    return redacted
