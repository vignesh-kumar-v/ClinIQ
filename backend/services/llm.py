import asyncio
from config import llm_client, CHAT_MODEL, MAX_HISTORY_TURNS
from logger import get_logger

log = get_logger("llm")

_SYSTEM = (
    "You are ClinIQ, a clinical memory assistant. You answer ONLY clinical and healthcare-related questions "
    "using the provided context. Be concise and clinically precise.\n\n"
    "CRITICAL RULES:\n"
    "1. If the answer is not in the context, say 'This information is not available in the patient records.'\n"
    "2. If the user asks a non-clinical question (coding, weather, recipes, trivia, jokes, etc.), "
    "refuse politely: 'I can only assist with clinical and healthcare-related questions.'\n"
    "3. Never generate code, poems, stories, or any non-medical content.\n"
    "4. Never hallucinate clinical values, lab results, or patient data."
)


def _build_messages(context: list[dict], history: list[dict], user_message: str) -> list[dict]:
    context_text = "\n".join(
        f"{i + 1}. [{item.get('source', '')}] {item['text']}"
        for i, item in enumerate(context)
    )
    system_content = f"{_SYSTEM}\n\nContext:\n{context_text}" if context_text else _SYSTEM
    return (
        [{"role": "system", "content": system_content}]
        + history[-MAX_HISTORY_TURNS:]
        + [{"role": "user", "content": user_message}]
    )


def _chat_sync(messages: list[dict]) -> str:
    response = llm_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content.strip()


async def chat(context: list[dict], history: list[dict], user_message: str) -> str:
    messages = _build_messages(context, history, user_message)
    log.info(f"LLM call model={CHAT_MODEL} context_chunks={len(context)} history_turns={len(history)}")
    log.debug(f"User message: {user_message[:120]!r}")
    try:
        answer = await asyncio.to_thread(_chat_sync, messages)
        log.info(f"LLM response ({len(answer)} chars): {answer[:120]!r}")
        return answer
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        raise


async def chat_stream(context: list[dict], history: list[dict], user_message: str):
    messages = _build_messages(context, history, user_message)
    log.info(f"LLM stream model={CHAT_MODEL} context_chunks={len(context)} history_turns={len(history)}")
    try:
        stream = llm_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
    except Exception as e:
        log.error(f"LLM stream failed: {e}")
        raise
