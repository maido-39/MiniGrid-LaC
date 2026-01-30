"""
Feedback prompt detection and prefix stripping.

Used by scenario_runner and other callers to decide if user input is feedback
and to extract the feedback text (e.g. strip "feedback :" prefix).
"""

from typing import Optional, List


# Default keywords: if any is contained in user input (case-insensitive), treat as feedback
FEEDBACK_KEYWORDS = [
    "wrong",
    "feedback :",
]


def is_feedback_prompt(
    text: Optional[str],
    keywords: Optional[List[str]] = None,
) -> bool:
    """
    Return True if text is considered a feedback prompt.
    Uses FEEDBACK_KEYWORDS by default; any keyword contained in text (case-insensitive) returns True.
    """
    if not text or not isinstance(text, str):
        return False
    kw = keywords if keywords is not None else FEEDBACK_KEYWORDS
    lower = text.lower()
    for keyword in kw:
        if keyword in lower:
            return True
    return False


def strip_feedback_prefix(
    text: str,
    prefix: str = "feedback :",
) -> str:
    """
    If text starts with prefix (case-insensitive), return text with that prefix stripped and trimmed.
    Otherwise return text unchanged.
    """
    if not text or not isinstance(text, str):
        return text or ""
    if text.lower().startswith(prefix.lower()):
        return text[len(prefix):].strip()
    return text
