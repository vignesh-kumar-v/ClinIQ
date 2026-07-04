from config import llm_client, CHAT_MODEL, MAX_HISTORY_TURNS
from logger import get_logger

log = get_logger("llm")

_SYSTEM = (
    "You are ClinIQ, a clinical memory assistant. Answer only using the provided context. "
    "Be concise and clinically precise. If the answer is not in the context, say so explicitly. "
    "Do not hallucinate clinical values."
)


async def chat(context: list[dict], history: list[dict], user_message: str) -> str:
    context_text = "\n".join(
        f"{i + 1}. [{item.get('source', '')}] {item['text']}"
        for i, item in enumerate(context)
    )
    system_content = f"{_SYSTEM}\n\nContext:\n{context_text}" if context_text else _SYSTEM

    messages = (
        [{"role": "system", "content": system_content}]
        + history[-MAX_HISTORY_TURNS:]
        + [{"role": "user", "content": user_message}]
    )

    log.info(f"LLM call model={CHAT_MODEL} context_chunks={len(context)} history_turns={len(history)}")
    log.debug(f"User message: {user_message[:120]!r}")
    try:
        response = llm_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
        )
        answer = response.choices[0].message.content.strip()
        log.info(f"LLM response ({len(answer)} chars): {answer[:120]!r}")
        return answer
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        raise
