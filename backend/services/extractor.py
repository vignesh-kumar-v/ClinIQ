import json
import re
from config import llm_client, INTENT_MODEL
from logger import get_logger

log = get_logger("extractor")

_PROMPT = """Extract structured clinical data from this input.
Return valid JSON only with these fields (no markdown, no explanation):
{{
  "action": "add_note | update_medication | add_condition | add_observation",
  "drug": string or null,
  "field": string or null,
  "value": string or null,
  "raw_note": string
}}
Input: {text}"""


async def extract_write_payload(text: str) -> dict:
    log.debug(f"Extracting write payload from: {text[:80]!r}")
    try:
        response = llm_client.chat.completions.create(
            model=INTENT_MODEL,
            messages=[{"role": "user", "content": _PROMPT.format(text=text)}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        log.debug(f"Raw extractor response: {raw[:200]!r}")
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        payload = json.loads(raw)
        log.info(f"Extracted action={payload.get('action')!r} drug={payload.get('drug')!r}")
        return payload
    except json.JSONDecodeError as e:
        log.error(f"JSON parse failed: {e} | raw={raw!r}")
        raise
    except Exception as e:
        log.error(f"Extraction failed: {e}")
        raise
