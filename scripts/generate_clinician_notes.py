#!/usr/bin/env python3
"""
Generate clinician notes from Synthea FHIR data.

Reads each patient's FHIR Bundle JSON, extracts their clinical history,
and generates realistic clinician notes (1-3 per patient) based on their
actual conditions, medications, lab results, and encounters.

Output: data/clinician_notes/{patient_id}.json
"""

import json
import os
import random
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

FHIR_DIR = Path("data/fhir")
OUTPUT_DIR = Path("data/clinician_notes")
PROGRESS_FILE = Path("data/clinician_notes_progress.txt")

random.seed(42)

NOTE_TEMPLATES = [
    {
        "type": "follow_up",
        "templates": [
            "Patient seen for follow-up of {condition}. {status_update}. "
            "Current medications: {medications}. "
            "Recent labs show {labs}. "
            "Plan: {plan}.",

            "Follow-up visit for {condition}. {symptom_status}. "
            "{medication_changes}. "
            "Labs reviewed: {labs}. "
            "Assessment: {assessment}. Plan: {plan}.",

            "Routine follow-up. {condition} remains {control_status}. "
            "{medication_note}. "
            "Vitals and labs: {labs}. "
            "Impression: {impression}. {plan}.",
        ],
    },
    {
        "type": "new_complaint",
        "templates": [
            "Patient presents with {symptom} for the past {duration}. "
            "History significant for {condition}. "
            "Current medications include {medications}. "
            "Physical exam: {exam_findings}. "
            "Assessment: {assessment}. Plan: {plan}.",

            "New complaint of {symptom} x {duration}. "
            "Background: {condition}. "
            "Medication review: {medications}. "
            "Exam notable for {exam_findings}. "
            "Impression: {impression}. {plan}.",

            "CC: {symptom}, duration {duration}. "
            "PMH: {condition}. "
            "Meds: {medications}. "
            "PE: {exam_findings}. "
            "A/P: {assessment}. {plan}.",
        ],
    },
    {
        "type": "medication_review",
        "templates": [
            "Medication review visit. Patient currently on {medications}. "
            "{medication_assessment}. "
            "Labs: {labs}. "
            "No acute complaints. {plan}.",

            "Annual medication reconciliation. Active prescriptions: {medications}. "
            "{medication_changes}. "
            "Recent labs: {labs}. "
            "Patient reports {symptom_status}. {plan}.",
        ],
    },
    {
        "type": "lab_review",
        "templates": [
            "Lab review visit. Recent results: {labs}. "
            "{lab_interpretation}. "
            "Patient with history of {condition}. "
            "Current medications: {medications}. "
            "Plan: {plan}.",

            "Reviewed lab results with patient. Notable findings: {labs}. "
            "{lab_interpretation}. "
            "Background: {condition}. "
            "Medication adjustments: {medication_changes}. "
            "Follow-up: {plan}.",
        ],
    },
    {
        "type": "annual_physical",
        "templates": [
            "Annual physical examination. Patient with history of {condition}. "
            "Current medications: {medications}. "
            "Vitals and screening labs: {labs}. "
            "Preventive care: {preventive}. "
            "Assessment: {assessment}. Plan: {plan}.",

            "Wellness visit. PMH significant for {condition}. "
            "Medication list reviewed: {medications}. "
            "Screening results: {labs}. "
            "Health maintenance: {preventive}. "
            "Impression: {impression}. {plan}.",
        ],
    },
]

STATUS_PHRASES = {
    "improving": [
        "significant improvement noted",
        "symptoms have improved since last visit",
        "condition is well-controlled",
        "doing well, symptoms resolved",
    ],
    "stable": [
        "condition is stable on current regimen",
        "no significant changes since last visit",
        "no new concerns reported",
        "status remains unchanged",
    ],
    "worsening": [
        "worsening symptoms reported",
        "condition has deteriorated since last visit",
        "symptoms are not well-controlled on current therapy",
        "experiencing increased frequency of symptoms",
    ],
    "new": [
        "this is a new finding",
        "first presentation of these symptoms",
        "no prior history of this complaint",
    ],
}

CONTROL_PHRASES = {
    "improving": ["well-controlled", "improving", "responding to treatment"],
    "stable": ["stable", "adequately managed", "at baseline"],
    "worsening": ["poorly controlled", "worsening", "not responding to current therapy"],
}

PLAN_PHRASES = [
    "Continue current management. Follow up in 3 months.",
    "Adjust medication dosage. Recheck labs in 4 weeks.",
    "Refer to specialist for further evaluation.",
    "Order additional labs including {extra_labs}. Follow up in 2 weeks.",
    "Start {new_med}. Titrate as tolerated. Return in 6 weeks.",
    "Continue monitoring. Return to clinic in 3 months for repeat labs.",
    "Increase {med} to {dose}. Reassess in 4 weeks.",
    "Discontinue {med}. Monitor for improvement. Follow up in 2 weeks.",
    "Add {new_med} to current regimen. Check labs in 1 month.",
    "No changes to current regimen. Routine follow-up in 6 months.",
]

EXAM_FINDINGS = [
    "mild tenderness on palpation",
    "no acute distress",
    "lungs clear bilaterally",
    "regular rate and rhythm",
    "abdomen soft, non-tender",
    "mild edema in lower extremities",
    "decreased range of motion",
    "normal gait and station",
    "no focal neurological deficits",
    "skin warm and dry",
    "mild erythema noted",
    "normal heart sounds, no murmurs",
]

PREVENTIVE_ITEMS = [
    "Immunizations up to date. Cancer screening discussed.",
    "Colonoscopy recommended. Mammogram ordered.",
    "Lipid panel and HbA1c ordered. Vaccines reviewed.",
    "Age-appropriate cancer screening discussed. Vaccinations current.",
    "Bone density scan recommended. Fall risk assessment completed.",
]

EXTRA_LABS_OPTIONS = [
    "CMP, CBC, lipid panel",
    "TSH, free T4",
    "HbA1c, fasting glucose",
    "BMP, LFTs",
    "CRP, ESR",
    "vitamin D, B12 levels",
    "iron studies, ferritin",
    "urinalysis, microalbumin",
]

NEW_MEDS = [
    "Metformin 500mg BID",
    "Lisinopril 10mg daily",
    "Atorvastatin 20mg daily",
    "Omeprazole 20mg daily",
    "Levothyroxine 50mcg daily",
    "Amlodipine 5mg daily",
    "Sertraline 50mg daily",
    "Gabapentin 300mg TID",
    "Ibuprofen 600mg TID prn",
    "Albuterol inhaler prn",
]


def load_patient_data(filepath: Path) -> dict[str, Any]:
    with open(filepath) as f:
        bundle = json.load(f)

    patient = {}
    conditions = []
    medications = []
    observations = []
    encounters = []

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        rt = resource.get("resourceType")

        if rt == "Patient":
            name = resource.get("name", [{}])[0]
            patient = {
                "id": resource["id"],
                "name": f"{name.get('given', [''])[0]} {name.get('family', '')}",
                "first_name": name.get("given", [""])[0],
                "last_name": name.get("family", ""),
                "gender": resource.get("gender", ""),
                "birth_date": resource.get("birthDate", ""),
            }

        elif rt == "Condition":
            code = resource.get("code", {}).get("coding", [{}])[0].get("display", "")
            onset = resource.get("onsetDateTime", "")
            if code:
                conditions.append({"code": code, "onset": onset})

        elif rt == "MedicationRequest":
            med = resource.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display", "")
            authored = resource.get("authoredOn", "")
            dosage = ""
            if resource.get("dosageInstruction"):
                dosage = resource["dosageInstruction"][0].get("text", "")
            if med:
                medications.append({"name": med, "date": authored, "dosage": dosage})

        elif rt == "Observation":
            code = resource.get("code", {}).get("coding", [{}])[0].get("display", "")
            code = code.replace("{score}", "").replace("{#}", "").strip()
            vq = resource.get("valueQuantity", {})
            value = vq.get("value")
            unit = vq.get("unit", "").replace("{score}", "").replace("{#}", "").strip()
            effective = resource.get("effectiveDateTime", "")
            if code and value is not None:
                observations.append({
                    "code": code,
                    "value": value,
                    "unit": unit,
                    "date": effective,
                })

        elif rt == "Encounter":
            enc_type = resource.get("type", [{}])[0].get("coding", [{}])[0].get("display", "")
            period = resource.get("period", {})
            start = period.get("start", "")
            reason = ""
            if resource.get("reasonCode"):
                reason = resource["reasonCode"][0].get("coding", [{}])[0].get("display", "")
            if enc_type:
                encounters.append({"type": enc_type, "date": start, "reason": reason})

    return {
        "patient": patient,
        "conditions": conditions,
        "medications": medications,
        "observations": observations,
        "encounters": encounters,
    }


def get_significant_conditions(conditions: list) -> list:
    """Filter out non-clinical conditions like employment status, education, etc."""
    skip_keywords = [
        "medication review due", "employment", "education", "social isolation",
        "stress", "never married", "married", "divorced", "widowed",
        "high school", "college", "part-time", "full-time", "retired",
        "patient", "finding", "situation",
    ]
    significant = []
    for c in conditions:
        code_lower = c["code"].lower()
        if not any(kw in code_lower for kw in skip_keywords):
            significant.append(c)
    return significant


def get_recent_observations(observations: list, n: int = 5) -> list:
    """Get the n most recent observations with values."""
    with_values = [o for o in observations if o["value"] is not None]
    with_values.sort(key=lambda o: o["date"], reverse=True)
    return with_values[:n]


def get_active_medications(medications: list) -> list:
    """Get unique medications sorted by most recent first."""
    seen = set()
    unique = []
    for m in sorted(medications, key=lambda m: m["date"], reverse=True):
        if m["name"] and m["name"] not in seen:
            seen.add(m["name"])
            unique.append(m)
    return unique


def format_medication_list(medications: list) -> str:
    if not medications:
        return "no active medications"
    names = [shorten_med_name(m["name"]) for m in medications[:5] if m["name"]]
    if not names:
        return "no active medications"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + ", and " + names[-1]


def shorten_med_name(name: str) -> str:
    """Extract the core drug name from verbose Synthea medication display names."""
    if not name:
        return name
    # Extract name from bracket suffix like "... [Flovent]"
    if " [" in name and name.endswith("]"):
        bracket = name.rfind("[")
        return name[bracket + 1:].rstrip("]")
    # Remove pack descriptions like "{24 (drospirenone ...) / 4 (inert ...) } Pack [Yaz 28 Day]"
    if "} Pack [" in name:
        bracket = name.rfind("[")
        if bracket > 0:
            return name[bracket + 1:].rstrip("]")
    # Remove NDA prefix like "NDA021457 200 ACTUAT ..."
    name = re.sub(r'^NDA\d+\s+', '', name)
    # Remove dosage forms to get just the drug name
    forms = [" Oral Tablet", " Oral Capsule", " Injectable Solution", " Transdermal System",
             " Extended Release Oral Tablet", " Mucosal Spray", " Inhalant Solution",
             " Oral Suspension", " Topical Cream", " Topical Ointment",
             " Chewable Tablet", " Metered Dose Inhaler", " Auto-Injector",
             " Disintegrating Oral Tablet", " Sublingual Tablet"]
    for form in forms:
        if form in name:
            name = name[:name.index(form)]
            break
    # Remove leading duration prefixes like "24 HR ", "168 HR "
    name = re.sub(r'^\d+\s*HR\s+', '', name)
    # Remove leading quantity prefixes like "120 ACTUAT ", "200 ACTUAT "
    name = re.sub(r'^\d+\s*ACTUAT\s+', '', name)
    return name.strip()


def format_lab_summary(observations: list) -> str:
    if not observations:
        return "no recent labs available"
    parts = []
    for o in observations[:4]:
        parts.append(f"{o['code']} {o['value']} {o['unit']}".strip())
    return "; ".join(parts) if parts else "no recent labs available"


def pick(items: list, default: str = "") -> str:
    return random.choice(items) if items else default


def generate_note(patient_data: dict, note_type: str) -> dict:
    p = patient_data["patient"]
    sig_conditions = get_significant_conditions(patient_data["conditions"])
    active_meds = get_active_medications(patient_data["medications"])
    recent_obs = get_recent_observations(patient_data["observations"])
    encounters = patient_data["encounters"]

    condition_name = sig_conditions[0]["code"] if sig_conditions else "no significant conditions"
    condition_list = ", ".join([c["code"] for c in sig_conditions[:3]]) if sig_conditions else "no chronic conditions"
    med_list = format_medication_list(active_meds)
    lab_summary = format_lab_summary(recent_obs)

    status = random.choice(["improving", "stable", "worsening"])
    status_text = pick(STATUS_PHRASES[status])
    control_text = pick(CONTROL_PHRASES[status])

    plan = pick(PLAN_PHRASES)
    plan = plan.replace("{extra_labs}", pick(EXTRA_LABS_OPTIONS))
    plan = plan.replace("{new_med}", pick(NEW_MEDS))
    if active_meds and active_meds[0]["name"]:
        plan = plan.replace("{med}", shorten_med_name(active_meds[0]["name"]))
        plan = plan.replace("{dose}", "a higher dose")
    else:
        plan = plan.replace("{med}", "current medication")
        plan = plan.replace("{dose}", "therapeutic dose")

    med_names = [shorten_med_name(m['name']) for m in active_meds if m['name']]
    medication_changes = "No changes to medication regimen"
    if med_names and random.random() > 0.7:
        medication_changes = f"Adjusted {random.choice(med_names)} dosage"

    medication_note = f"Currently on {med_list}" if active_meds else "No active medications"
    medication_assessment = "All medications reviewed, appropriate for current conditions" if random.random() > 0.3 else "Medication regimen optimized"

    template_group = next((g for g in NOTE_TEMPLATES if g["type"] == note_type), NOTE_TEMPLATES[0])
    template = pick(template_group["templates"])

    note_text = template.format(
        condition=condition_name,
        condition_list=condition_list,
        status_update=status_text,
        symptom_status=status_text,
        control_status=control_text,
        medications=med_list,
        medication_note=medication_note,
        medication_changes=medication_changes,
        medication_assessment=medication_assessment,
        labs=lab_summary,
        lab_interpretation="Results are within expected range" if status != "worsening" else "Results show concerning trends requiring intervention",
        assessment=f"{condition_name} - {status}" if status != "new" else f"New onset {condition_name}",
        impression=f"{condition_name} is {control_text}" if status != "new" else f"New diagnosis of {condition_name}",
        plan=plan,
        symptom=pick(["fatigue", "headache", "shortness of breath", "joint pain", "dizziness", "nausea", "chest discomfort", "back pain"]),
        duration=pick(["3 days", "1 week", "2 weeks", "5 days", "10 days"]),
        exam_findings=pick(EXAM_FINDINGS),
        preventive=pick(PREVENTIVE_ITEMS),
    )

    last_encounter_date = encounters[-1]["date"] if encounters else datetime.now().isoformat()
    try:
        note_date = datetime.fromisoformat(last_encounter_date.replace("Z", "+00:00"))
        note_date += timedelta(hours=random.randint(1, 48))
    except (ValueError, TypeError):
        note_date = datetime.now()

    return {
        "note_id": str(uuid.uuid4()),
        "patient_id": p["id"],
        "patient_name": p["name"],
        "timestamp": note_date.isoformat() + "Z",
        "note_type": note_type,
        "note_text": note_text,
        "source": "clinician_note",
    }


def generate_notes_for_patient(patient_data: dict) -> list:
    """Generate 1-3 notes per patient based on data richness."""
    sig_conditions = get_significant_conditions(patient_data["conditions"])
    encounters = patient_data["encounters"]
    observations = patient_data["observations"]

    num_notes = 1
    if len(sig_conditions) >= 3 and len(encounters) >= 10:
        num_notes = 3
    elif len(sig_conditions) >= 2 and len(encounters) >= 5:
        num_notes = 2

    note_types = ["follow_up", "medication_review", "lab_review", "annual_physical"]
    if random.random() < 0.2:
        note_types.append("new_complaint")

    selected_types = random.sample(note_types, min(num_notes, len(note_types)))

    notes = []
    for nt in selected_types:
        note = generate_note(patient_data, nt)
        notes.append(note)

    return notes


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fhir_files = sorted(FHIR_DIR.glob("*.json"))
    total = len(fhir_files)
    print(f"Found {total} FHIR files in {FHIR_DIR}")

    completed = set()
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            completed = set(line.strip() for line in f)
        print(f"Resuming: {len(completed)} already processed")

    all_notes = []
    for i, fhir_path in enumerate(fhir_files):
        patient_id = fhir_path.stem
        if patient_id in completed:
            continue

        try:
            patient_data = load_patient_data(fhir_path)
            notes = generate_notes_for_patient(patient_data)

            output_path = OUTPUT_DIR / f"{patient_data['patient']['id']}.json"
            with open(output_path, "w") as f:
                json.dump(notes, f, indent=2)

            all_notes.extend(notes)
            completed.add(patient_id)

            with open(PROGRESS_FILE, "a") as f:
                f.write(f"{patient_id}\n")

            if (i + 1) % 500 == 0:
                print(f"  Progress: {i + 1}/{total} ({len(all_notes)} notes generated)")

        except Exception as e:
            print(f"  ERROR processing {fhir_path.name}: {e}")
            continue

    print(f"\nDone! Generated {len(all_notes)} clinician notes for {len(completed)} patients.")
    print(f"Output: {OUTPUT_DIR}/")
    print(f"Progress file: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
