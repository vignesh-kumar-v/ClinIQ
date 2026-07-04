import asyncio
from config import synthea_col, mtsamples_col, TOP_K_SYNTHEA, TOP_K_MTSAMPLES
from logger import get_logger

log = get_logger("retriever")


def _synthea_search(query_vector: list[float], patient_id: str) -> list[dict]:
    log.debug(f"Synthea search patient_id={patient_id} top_k={TOP_K_SYNTHEA}")
    try:
        results = synthea_col.query(
            query_embeddings=[query_vector],
            n_results=TOP_K_SYNTHEA,
            where={"patient_id": {"$eq": patient_id}},
            include=["documents", "metadatas"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        log.debug(f"Synthea returned {len(docs)} results")
        return [
            {
                "text": doc,
                "data_type": m.get("data_type", ""),
                "date": m.get("date", ""),
                "source": "synthea_structured",
            }
            for doc, m in zip(docs, metas)
        ]
    except Exception as e:
        log.warning(f"Synthea search failed for patient {patient_id}: {e}")
        return []


def _mtsamples_search(query_vector: list[float]) -> list[dict]:
    log.debug(f"MTSamples search top_k={TOP_K_MTSAMPLES}")
    try:
        results = mtsamples_col.query(
            query_embeddings=[query_vector],
            n_results=TOP_K_MTSAMPLES,
            include=["documents", "metadatas"],
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        log.debug(f"MTSamples returned {len(docs)} results")
        return [
            {
                "text": doc,
                "specialty": m.get("specialty", ""),
                "section": m.get("section", ""),
                "source": "mtsamples_knowledge",
            }
            for doc, m in zip(docs, metas)
        ]
    except Exception as e:
        log.warning(f"MTSamples search failed: {e}")
        return []


async def search(query_vector: list[float], patient_id: str) -> list[dict]:
    # query_vector must already be embedded with is_query=True (prompt_name="query")
    log.info(f"Parallel retrieval for patient={patient_id}")
    synthea_results, mtsamples_results = await asyncio.gather(
        asyncio.to_thread(_synthea_search, query_vector, patient_id),
        asyncio.to_thread(_mtsamples_search, query_vector),
    )
    merged = synthea_results + mtsamples_results
    seen = set()
    unique = []
    for item in merged:
        t = item["text"]
        if t not in seen:
            seen.add(t)
            unique.append(item)
    log.info(f"Retrieval complete: {len(synthea_results)} synthea + {len(mtsamples_results)} mtsamples = {len(unique)} unique")
    return unique
