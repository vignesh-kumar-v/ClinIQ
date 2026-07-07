import json
import os
from logger import get_logger

log = get_logger("encounter_enricher")

_encounter_map: dict[tuple, str] = {}
_loaded = False

FHIR_DIR = os.environ.get("FHIR_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data", "fhir"))


def load_encounter_types():
    global _encounter_map, _loaded
    if _loaded:
        return
    log.info("Loading encounter type map from FHIR data...")
    count = 0
    for fname in os.listdir(FHIR_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(FHIR_DIR, fname)) as f:
                data = json.load(f)
        except Exception:
            continue
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
            if enc_type:
                _encounter_map[(patient_id, enc_date)] = enc_type
                count += 1
    _loaded = True
    log.info(f"Loaded {count} encounter type mappings")


def enrich_encounter(patient_id: str, text: str, metadata: dict) -> str:
    if "unspecified" not in text:
        return text
    date = metadata.get("date", "")
    enc_type = _encounter_map.get((patient_id, date), "")
    if enc_type:
        return f"Visit on {date}, reason: {enc_type}"
    return text
