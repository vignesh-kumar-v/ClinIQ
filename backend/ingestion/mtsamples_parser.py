import re

SECTION_HEADERS = [
    "SUBJECTIVE",
    "OBJECTIVE",
    "ASSESSMENT",
    "PLAN",
    "HPI",
    "CHIEF COMPLAINT",
    "HISTORY",
    "PHYSICAL EXAMINATION",
    "LABORATORY",
    "IMPRESSION",
]

# Regex that matches any header at start of a line (case-insensitive)
_HEADER_PATTERN = re.compile(
    r"^(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")[:\s]*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_transcription(text: str, specialty: str) -> list[dict]:
    chunks = []
    matches = list(_HEADER_PATTERN.finditer(text))

    if not matches:
        stripped = text.strip()
        if stripped:
            chunks.append({
                "text": stripped,
                "metadata": {"specialty": specialty, "section": "full", "source": "mtsamples", "keywords": ""},
            })
        return chunks

    boundaries = [(m.start(), m.group(1).upper()) for m in matches]

    for i, (start, section) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        # Skip the header line itself
        body_start = text.index("\n", start) + 1 if "\n" in text[start:end] else end
        body = text[body_start:end].strip()
        if body:
            chunks.append({
                "text": body,
                "metadata": {"specialty": specialty, "section": section, "source": "mtsamples", "keywords": ""},
            })

    return chunks
