import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import chroma_client, synthea_col

FHIR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fhir")


def build_encounter_map():
    enc_map = {}
    for fname in os.listdir(FHIR_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(FHIR_DIR, fname)) as f:
            data = json.load(f)
        patient_id = None
        for entry in data.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                patient_id = resource.get("id", "")
                break
        if not patient_id:
            continue
        for entry in data.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "Encounter":
                continue
            period = resource.get("period", {})
            enc_date = period.get("start", "")[:10]
            if not enc_date:
                continue
            type_list = resource.get("type", [])
            enc_type = ""
            if type_list:
                enc_type = type_list[0].get("text", "") or (type_list[0].get("coding", [{}])[0].get("display", ""))
            key = (patient_id, enc_date)
            if key not in enc_map:
                enc_map[key] = enc_type
    return enc_map


def main():
    print("Building encounter type map from FHIR data...")
    enc_map = build_encounter_map()
    print(f"Loaded {len(enc_map)} encounter (patient_id, date) -> type mappings")

    print("Fetching encounter chunks from ChromaDB in batches...")
    offset = 0
    batch_size = 10000
    updated = 0
    total_processed = 0

    while True:
        results = synthea_col.get(
            where={"data_type": {"$eq": "encounter"}},
            include=["documents", "metadatas", "embeddings"],
            limit=batch_size,
            offset=offset,
        )
        if not results["ids"]:
            break

        batch_ids = []
        batch_docs = []
        batch_metas = []
        batch_embs = []

        for chunk_id, doc, meta, emb in zip(results["ids"], results["documents"], results["metadatas"], results["embeddings"]):
            total_processed += 1
            if "unspecified" not in doc:
                continue

            pid = meta.get("patient_id", "")
            date = meta.get("date", "")
            enc_type = enc_map.get((pid, date), "")

            if not enc_type:
                continue

            new_text = f"Visit on {date}, reason: {enc_type}"
            meta["encounter_type"] = enc_type
            meta["encounter_date"] = date

            batch_ids.append(chunk_id)
            batch_docs.append(new_text)
            batch_metas.append(meta)
            batch_embs.append(emb)
            updated += 1

            if len(batch_ids) >= 500:
                synthea_col.update(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=batch_embs)
                print(f"  Updated {updated} so far (processed {total_processed})...")
                batch_ids.clear()
                batch_docs.clear()
                batch_metas.clear()
                batch_embs.clear()

        if batch_ids:
            synthea_col.update(ids=batch_ids, documents=batch_docs, metadatas=batch_metas, embeddings=batch_embs)

        offset += batch_size
        print(f"  Batch done. Processed {total_processed} so far, updated {updated}...")

    print(f"Done. Updated {updated} encounter chunks out of {total_processed} total.")


if __name__ == "__main__":
    main()
