import re

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)",
    r"you\s+are\s+now\s+(DAN|jailbroken|unrestricted)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"forget\s+(your|all)\s+(instructions?|training|rules?)",
    r"system\s*:\s*",
    r"assistant\s*:\s*",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"\[system\]",
    r"\[assistant\]",
    r"\[/INST\]",
    r"\[INST\]",
    r"</?system>",
    r"</?assistant>",
    r"</?instruction>",
    r"---+\s*(system|assistant|instruction)",
    r"new\s+system\s+prompt",
    r"override\s+(system|prompt|instructions?)",
    r"bypass\s+(filter|guardrail|restriction)",
    r"disregard\s+(previous|all)\s+(instructions?|constraints?)",
    r"act\s+as\s+(if\s+you\s+are|a\s+different)",
    r"you\s+must\s+(ignore|disregard|forget)",
    r"do\s+not\s+follow\s+(your|the)\s+(instructions?|rules?)",
    r"switch\s+(roles?|personas?)",
    r"from\s+now\s+on\s+you\s+are",
]

_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_MAX_INPUT_LENGTH = 4000


def sanitize(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip()[: _MAX_INPUT_LENGTH]

    text = _INJECTION_RE.sub("[filtered]", text)

    text = text.replace("\x00", "")

    return text
