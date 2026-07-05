from datetime import date, datetime


def _calc_age(birth_date_str: str) -> int:
    try:
        born = datetime.strptime(birth_date_str[:10], "%Y-%m-%d").date()
        today = date.today()
        return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    except Exception:
        return 0


def _resolve_encounter_ref(ref: str) -> str:
    if ref.startswith("urn:uuid:"):
        return ref.split(":")[-1]
    return ""


def parse_patient(fhir_json: dict) -> list[dict]:
    chunks = []
    entries = fhir_json.get("entry", [])

    patient_id = None
    patient_name = None

    encounters: dict[str, dict] = {}

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Encounter":
            enc_id = resource.get("id", "")
            if not enc_id:
                continue
            type_list = resource.get("type", [])
            enc_type = ""
            if type_list:
                enc_type = type_list[0].get("text", "") or (type_list[0].get("coding", [{}])[0].get("display", ""))
            period = resource.get("period", {})
            enc_date = period.get("start", "")[:10]
            encounters[enc_id] = {"type": enc_type, "date": enc_date}

    for entry in entries:
        resource = entry.get("resource", {})
        if resource.get("resourceType") != "Patient":
            continue
        patient_id = resource.get("id", "")
        name_block = resource.get("name", [{}])[0]
        given = " ".join(name_block.get("given", []))
        family = name_block.get("family", "")
        patient_name = f"{given} {family}".strip()
        gender = resource.get("gender", "unknown")
        birth_date = resource.get("birthDate", "")
        age = _calc_age(birth_date) if birth_date else "unknown"
        chunks.append({
            "text": f"Patient: {patient_name}, {age}y, {gender}",
            "metadata": {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "data_type": "demographics",
                "source": "synthea",
                "date": birth_date,
            },
        })
        break

    if not patient_id or not patient_name:
        return []

    base_meta = {"patient_id": patient_id, "patient_name": patient_name, "source": "synthea"}

    for entry in entries:
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "")

        if rtype == "Condition":
            desc = (
                resource.get("code", {}).get("text")
                or (resource.get("code", {}).get("coding") or [{}])[0].get("display", "")
            )
            onset = resource.get("onsetDateTime", resource.get("onsetPeriod", {}).get("start", ""))
            if not desc:
                continue
            enc_ref = _resolve_encounter_ref((resource.get("encounter", {}) or {}).get("reference", ""))
            enc = encounters.get(enc_ref, {})
            enc_context = f" during {enc['type']}" if enc.get("type") else ""
            chunks.append({
                "text": f"{desc} onset {onset[:10] if onset else 'unknown'}{enc_context}",
                "metadata": {
                    **base_meta,
                    "data_type": "condition",
                    "date": onset[:10] if onset else "",
                    "encounter_type": enc.get("type", ""),
                    "encounter_date": enc.get("date", ""),
                },
            })

        elif rtype == "MedicationRequest":
            med_text = (
                resource.get("medicationCodeableConcept", {}).get("text")
                or (resource.get("medicationCodeableConcept", {}).get("coding") or [{}])[0].get("display", "")
            )
            if not med_text:
                continue
            dosage = ""
            dosage_list = resource.get("dosageInstruction", [])
            if dosage_list:
                dosage = dosage_list[0].get("text", "")
            authored = resource.get("authoredOn", "")
            enc_ref = _resolve_encounter_ref((resource.get("encounter", {}) or {}).get("reference", ""))
            enc = encounters.get(enc_ref, {})
            enc_context = f" during {enc['type']}" if enc.get("type") else ""
            chunks.append({
                "text": f"{med_text} {dosage} started {authored[:10] if authored else 'unknown'}{enc_context}".strip(),
                "metadata": {
                    **base_meta,
                    "data_type": "medication",
                    "date": authored[:10] if authored else "",
                    "drug": med_text,
                    "encounter_type": enc.get("type", ""),
                    "encounter_date": enc.get("date", ""),
                },
            })

        elif rtype == "Observation":
            obs_text = (
                resource.get("code", {}).get("text")
                or (resource.get("code", {}).get("coding") or [{}])[0].get("display", "")
            )
            if not obs_text:
                continue
            value = ""
            if "valueQuantity" in resource:
                vq = resource["valueQuantity"]
                value = f"{vq.get('value', '')} {vq.get('unit', '')}".strip()
            elif "valueString" in resource:
                value = resource["valueString"]
            elif "valueCodeableConcept" in resource:
                value = resource["valueCodeableConcept"].get("text", "")
            obs_date = resource.get("effectiveDateTime", "")[:10]
            if not value:
                continue
            enc_ref = _resolve_encounter_ref((resource.get("encounter", {}) or {}).get("reference", ""))
            enc = encounters.get(enc_ref, {})
            enc_context = f" during {enc['type']}" if enc.get("type") else ""
            chunks.append({
                "text": f"{obs_text}: {value} on {obs_date or 'unknown'}{enc_context}",
                "metadata": {
                    **base_meta,
                    "data_type": "observation",
                    "date": obs_date,
                    "encounter_type": enc.get("type", ""),
                    "encounter_date": enc.get("date", ""),
                },
            })

        elif rtype == "Encounter":
            enc_date = resource.get("period", {}).get("start", "")[:10]
            reason_list = resource.get("reasonCode", [])
            reason = ""
            if reason_list:
                reason = (
                    reason_list[0].get("text")
                    or (reason_list[0].get("coding") or [{}])[0].get("display", "")
                )
            if not enc_date:
                continue
            chunks.append({
                "text": f"Visit on {enc_date}, reason: {reason or 'unspecified'}",
                "metadata": {**base_meta, "data_type": "encounter", "date": enc_date},
            })

    return chunks
