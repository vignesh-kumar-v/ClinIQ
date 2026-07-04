from config import llm_client, INTENT_MODEL
from logger import get_logger

log = get_logger("intent")

_PROMPT = """Classify this clinical input as read, write, or mixed.
write = contains new clinical data, updates, prescriptions, visit notes, changes
read = question, summary request, lookup
mixed = contains both new data AND a question
Return only one word: read, write, or mixed.
Input: {text}"""


async def classify_intent(text: str) -> str:
    log.debug(f"Classifying intent for: {text[:80]!r}")
    try:
        response = llm_client.chat.completions.create(
            model=INTENT_MODEL,
            messages=[{"role": "user", "content": _PROMPT.format(text=text)}],
            temperature=0,
        )
        result = response.choices[0].message.content.strip().lower()
        log.info(f"Intent={result!r} for input: {text[:60]!r}")
        return result
    except Exception as e:
        log.error(f"Intent classification failed: {e}")
        raise
