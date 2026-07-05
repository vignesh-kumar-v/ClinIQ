from config import llm_client, INTENT_MODEL
from logger import get_logger
from services.sanitizer import sanitize

log = get_logger("intent")

_PROMPT = """Classify this input as read, write, mixed, or out_of_scope.
write = contains new clinical data, updates, prescriptions, visit notes, changes
read = clinical question, summary request, patient lookup, medical knowledge query
mixed = contains both new clinical data AND a clinical question
out_of_scope = not related to healthcare, medicine, or clinical work (e.g. coding, weather, recipes, general trivia, jokes, personal questions)
Return only one word: read, write, mixed, or out_of_scope.
Input: {text}"""


async def classify_intent(text: str) -> str:
    text = sanitize(text)
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
