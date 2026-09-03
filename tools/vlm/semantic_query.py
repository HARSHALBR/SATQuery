"""Convert temporal SATQuery questions into static semantic VLM questions."""

import re


_TEMPORAL_PATTERNS = (
    r"\bbetween\b",
    r"\bfrom\b",
    r"\bto\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\bchange\b",
    r"\bchanged\b",
    r"\bincrease\b",
    r"\bincreased\b",
    r"\bdecrease\b",
    r"\bdecreased\b",
    r"\bloss\b",
    r"\bgrowth\b",
    r"\bexpanded\b",
    r"\bexpansion\b",
)


def build_semantic_query(query: str) -> str:
    """
    Convert a temporal/change query into a static scene interpretation query.

    RS-InternVL is trained for static semantic VQA rather than temporal
    change detection, so the returned question must describe one observation.
    """
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    text = query.strip()
    lower = text.lower()

    if any(re.search(pattern, lower) for pattern in _TEMPORAL_PATTERNS):
        subject = None
        if "vegetation" in lower or "forest" in lower:
            subject = "vegetation or forest"
        elif "urban" in lower or "built-up" in lower or "built up" in lower:
            subject = "built-up or urban features"
        elif "water" in lower:
            subject = "water"
        elif "agriculture" in lower or "crop" in lower or "farm" in lower:
            subject = "agriculture or crops"
        elif "land-cover" in lower or "land cover" in lower:
            subject = "land-cover features"
        
        if subject:
            return f"Is {subject} present in this satellite patch?"
        else:
            return "What land-cover features are visible in this satellite patch?"

    return text
