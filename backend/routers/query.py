from fastapi import APIRouter, HTTPException
from models import QueryRequest, QueryResponse
from services import intent, extractor, embedder, retriever, llm, writer
from logger import get_logger

log = get_logger("router.query")
router = APIRouter()


@router.get("/patients/{patient_id}/chat-history")
async def get_chat_history(patient_id: str):
    history = await writer.get_chat_history(patient_id)
    return {"patient_id": patient_id, "history": history}


@router.post("/patients/{patient_id}/query", response_model=QueryResponse)
async def query_patient(patient_id: str, body: QueryRequest):
    log.info(f"Query patient={patient_id} session={body.session_id} message={body.message[:80]!r}")

    detected_intent = await intent.classify_intent(body.message)

    if detected_intent == "out_of_scope":
        log.info(f"Out-of-scope query blocked for patient={patient_id}")
        refusal = "I'm a clinical memory assistant. I can only help with healthcare-related questions about patient records, medical knowledge, and clinical documentation. Please ask a clinical question."
        await writer.save_chat_turn(patient_id, body.session_id, "user", body.message)
        await writer.save_chat_turn(patient_id, body.session_id, "assistant", refusal)
        return QueryResponse(answer=refusal, intent="out_of_scope", sources=[])

    if detected_intent == "write":
        log.info(f"Write path triggered for patient={patient_id}")
        payload = await extractor.extract_write_payload(body.message)
        patient_name = _get_patient_name(patient_id)
        await _dispatch_write(payload, patient_id, patient_name, body.message)
        await writer.log_activity(patient_id, payload.get("action", "write"), body.message[:80])
        confirmation = f"Recorded: {payload.get('action', 'update')} for patient."
        await writer.save_chat_turn(patient_id, body.session_id, "user", body.message)
        await writer.save_chat_turn(patient_id, body.session_id, "assistant", confirmation)
        return QueryResponse(answer=confirmation, intent="write", sources=[])

    if detected_intent == "read":
        log.info(f"Read path triggered for patient={patient_id}")
        history = await writer.get_chat_history(patient_id)
        query_vector = embedder.embed(body.message, is_query=True)
        context = await retriever.search(query_vector, patient_id)
        answer = await llm.chat(context, history, body.message)
        await writer.save_chat_turn(patient_id, body.session_id, "user", body.message)
        await writer.save_chat_turn(patient_id, body.session_id, "assistant", answer)
        sources = [c.get("source", "") for c in context]
        return QueryResponse(answer=answer, intent="read", sources=sources)

    # mixed
    log.info(f"Mixed path triggered for patient={patient_id}")
    payload = await extractor.extract_write_payload(body.message)
    patient_name = _get_patient_name(patient_id)
    await _dispatch_write(payload, patient_id, patient_name, body.message)
    await writer.log_activity(patient_id, payload.get("action", "write"), body.message[:80])

    history = await writer.get_chat_history(patient_id)
    query_vector = embedder.embed(body.message, is_query=True)
    context = await retriever.search(query_vector, patient_id)
    answer = await llm.chat(context, history, body.message)
    await writer.save_chat_turn(patient_id, body.session_id, "user", body.message)
    await writer.save_chat_turn(patient_id, body.session_id, "assistant", answer)
    sources = [c.get("source", "") for c in context]
    return QueryResponse(answer=answer, intent="mixed", sources=sources)


def _get_patient_name(patient_id: str) -> str:
    from config import patient_index_col
    try:
        results = patient_index_col.get(
            where={"patient_id": {"$eq": patient_id}},
            include=["metadatas"],
        )
        if results["metadatas"]:
            return results["metadatas"][0].get("patient_name", "Unknown")
    except Exception as e:
        log.warning(f"Could not fetch patient_name for {patient_id}: {e}")
    return "Unknown"


async def _dispatch_write(payload: dict, patient_id: str, patient_name: str, raw_text: str) -> None:
    action = payload.get("action", "")
    log.info(f"Dispatching write action={action!r} patient={patient_id}")
    if action == "add_note":
        vec = embedder.embed(raw_text)
        await writer.add_note(patient_id, patient_name, raw_text, "follow_up", vec)
    elif action == "update_medication":
        drug = payload.get("drug") or ""
        field = payload.get("field") or ""
        value = payload.get("value") or ""
        if drug and field and value:
            await writer.update_medication(patient_id, drug, field, value)
        else:
            log.warning(f"update_medication missing fields: drug={drug!r} field={field!r} value={value!r}")
    elif action == "add_observation":
        vec = embedder.embed(raw_text)
        await writer.add_observation(patient_id, patient_name, raw_text, vec)
    elif action == "add_condition":
        vec = embedder.embed(raw_text)
        await writer.add_observation(patient_id, patient_name, raw_text, vec)
    else:
        log.warning(f"Unknown write action={action!r}")
