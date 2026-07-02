"""
Master ingestion script. Runs everything in order:
  1. Normalize section names in SQLite (mtsamples)
  2. Embed mtsamples chunks → ChromaDB: mtsamples_chunks
  3. Embed FHIR patients + clinician notes → ChromaDB: synthea_structured, patient_index
  4. Setup logs DB (activity_log, audit_log)

Usage:
    python ingest.py [--model qwen3-embedding:8b] [--batch-size 16] [--reset]
    python ingest.py --only mtsamples
    python ingest.py --only patients
    python ingest.py --only logs

Prerequisites:
    pip install chromadb ollama tqdm
    ollama pull qwen3-embedding:8b
"""

import argparse
import base64
import json
import sqlite3
import sys
from pathlib import Path

import chromadb
import ollama
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DB_PATH = ROOT / "data" / "mtsamples_staging.db"
FHIR_PATH = ROOT / "data" / "fhir"
NOTES_PATH = ROOT / "data" / "clinician_notes"
CHROMA_PATH = ROOT / "chroma_db"
LOGS_DB = ROOT / "data" / "logs.db"

MAX_CHARS = 3000


# ── text utils ────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    text = text.lstrip(",").lstrip(" ")
    return text[:MAX_CHARS] if len(text) > MAX_CHARS else text


# ── section normalization ─────────────────────────────────────────────────────

SECTION_MAP = {
    # HISTORY OF PRESENT ILLNESS
    "HPI": "HISTORY OF PRESENT ILLNESS",
    "HISTORY": "HISTORY OF PRESENT ILLNESS",
    "PRESENT ILLNESS": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF ILLNESS": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF PRESENT COMPLAINT": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF PRESENT PROBLEM": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF PRESENT INJURY": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF PRESENTING COMPLAINT": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF PRESENTING ILLNESS": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF PRESENTING PROBLEM": "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF THE PRESENT ILLNESS": "HISTORY OF PRESENT ILLNESS",
    "BRIEF HISTORY OF PRESENT ILLNESS": "HISTORY OF PRESENT ILLNESS",
    "CURRENT HISTORY OF PRESENT ILLNESS": "HISTORY OF PRESENT ILLNESS",
    "CURRENT HISTORY": "HISTORY OF PRESENT ILLNESS",
    "INTERVAL HISTORY": "HISTORY OF PRESENT ILLNESS",
    "INTERIM HISTORY": "HISTORY OF PRESENT ILLNESS",
    "PRESENT COMPLAINTS": "HISTORY OF PRESENT ILLNESS",
    "PRESENT PROBLEMS": "HISTORY OF PRESENT ILLNESS",
    "PRESENTING PROBLEM": "HISTORY OF PRESENT ILLNESS",
    "PRESENTING PROBLEMS": "HISTORY OF PRESENT ILLNESS",
    "BRIEF HISTORY": "HISTORY OF PRESENT ILLNESS",
    "CLINICAL HISTORY": "HISTORY OF PRESENT ILLNESS",
    # PAST MEDICAL HISTORY
    "PMH": "PAST MEDICAL HISTORY",
    "PAST MEDICAL HX": "PAST MEDICAL HISTORY",
    "PAST HISTORY": "PAST MEDICAL HISTORY",
    "PREVIOUS MEDICAL HISTORY": "PAST MEDICAL HISTORY",
    "PRIOR MEDICAL HISTORY": "PAST MEDICAL HISTORY",
    "PERTINENT MEDICAL HISTORY": "PAST MEDICAL HISTORY",
    "MEDICAL HISTORY": "PAST MEDICAL HISTORY",
    "ADULT MEDICAL PROBLEMS": "PAST MEDICAL HISTORY",
    "PAST MEDICAL CONDITIONS": "PAST MEDICAL HISTORY",
    "PRIMARY MEDICAL HISTORY": "PAST MEDICAL HISTORY",
    "PATIENT HISTORY": "PAST MEDICAL HISTORY",
    "PAST MEDICAL AND SURGICAL HISTORY": "PAST MEDICAL HISTORY",
    "PAST MEDICAL/SURGICAL HISTORY": "PAST MEDICAL HISTORY",
    "SIGNIFICANT PAST MEDICAL AND SURGICAL HISTORY": "PAST MEDICAL HISTORY",
    "OTHER SIGNIFICANT MEDICAL HISTORY/SURGERIES": "PAST MEDICAL HISTORY",
    "PAST MEDICAL HISTORY/SURGERIES/HOSPITALIZATIONS": "PAST MEDICAL HISTORY",
    "PAST MEDICAL HISTORY / SURGERY / HOSPITALIZATIONS": "PAST MEDICAL HISTORY",
    # PAST SURGICAL HISTORY
    "PSH": "PAST SURGICAL HISTORY",
    "PAST SURGICAL HX": "PAST SURGICAL HISTORY",
    "SURGICAL HISTORY": "PAST SURGICAL HISTORY",
    "PREVIOUS SURGICAL HISTORY": "PAST SURGICAL HISTORY",
    "PREVIOUS SURGERIES": "PAST SURGICAL HISTORY",
    "PRIOR SURGERIES": "PAST SURGICAL HISTORY",
    "PRIOR SURGERIES AND INTERVENTIONS": "PAST SURGICAL HISTORY",
    "PAST SURGERIES": "PAST SURGICAL HISTORY",
    "PREVIOUS OPERATIONS": "PAST SURGICAL HISTORY",
    # SOCIAL HISTORY
    "SHX": "SOCIAL HISTORY",
    "SOCIAL HX": "SOCIAL HISTORY",
    "SOCIAL": "SOCIAL HISTORY",
    "SOCH": "SOCIAL HISTORY",
    "SOCIAL HABITS": "SOCIAL HISTORY",
    "SOCIAL BACKGROUND": "SOCIAL HISTORY",
    "PERSONAL/SOCIAL HISTORY": "SOCIAL HISTORY",
    "PERSONAL AND SOCIAL HISTORY": "SOCIAL HISTORY",
    "PERSONAL HISTORY": "SOCIAL HISTORY",
    "SOCIOECONOMIC HISTORY": "SOCIAL HISTORY",
    "SOCIO-ECONOMIC HISTORY": "SOCIAL HISTORY",
    # FAMILY HISTORY
    "FHX": "FAMILY HISTORY",
    "FAMILY HX": "FAMILY HISTORY",
    "FAMILY MEDICAL HISTORY": "FAMILY HISTORY",
    "FAMILY BACKGROUND": "FAMILY HISTORY",
    "FAMILY PSYCHIATRIC HISTORY": "FAMILY HISTORY",
    # MEDICATIONS
    "MEDS": "MEDICATIONS",
    "CURRENT MEDICATIONS": "MEDICATIONS",
    "CURRENT MEDS": "MEDICATIONS",
    "CURRENT MEDICATION": "MEDICATIONS",
    "CURRENT MEDICINES": "MEDICATIONS",
    "HOME MEDICATIONS": "MEDICATIONS",
    "MEDICATIONS AT HOME": "MEDICATIONS",
    "CURRENT HOME MEDICATIONS": "MEDICATIONS",
    "MEDICATION": "MEDICATIONS",
    "MEDICINES": "MEDICATIONS",
    "PRESENT MEDICATIONS": "MEDICATIONS",
    "OUTPATIENT MEDICATIONS": "MEDICATIONS",
    "INPATIENT MEDICATIONS": "MEDICATIONS",
    "MEDICATIONS OUTPATIENT": "MEDICATIONS",
    "MEDICATIONS INPATIENT": "MEDICATIONS",
    "MEDICATION HISTORY": "MEDICATIONS",
    "DRUG HISTORY": "MEDICATIONS",
    "OTHER MEDICATIONS": "MEDICATIONS",
    "NEW MEDICATIONS": "MEDICATIONS",
    "MEDS ON ADMISSION": "MEDICATIONS",
    "MEDICATIONS ON ADMISSION": "MEDICATIONS",
    "MEDICATIONS PRIOR TO ADMISSION": "MEDICATIONS",
    # ALLERGIES
    "DRUG ALLERGIES": "ALLERGIES",
    "MEDICATION ALLERGIES": "ALLERGIES",
    "ALLERGIES TO MEDICATIONS": "ALLERGIES",
    "ALLERGIC": "ALLERGIES",
    # PHYSICAL EXAMINATION
    "PHYSICAL EXAM": "PHYSICAL EXAMINATION",
    "EXAM": "PHYSICAL EXAMINATION",
    "EXAMINATION": "PHYSICAL EXAMINATION",
    "PEX": "PHYSICAL EXAMINATION",
    "CLINICAL/PHYSICAL EXAMINATION": "PHYSICAL EXAMINATION",
    "ADMISSION PHYSICAL": "PHYSICAL EXAMINATION",
    "ADMIT EXAM": "PHYSICAL EXAMINATION",
    # VITAL SIGNS
    "VITALS": "VITAL SIGNS",
    "VITAL SIGNS/GENERAL": "VITAL SIGNS",
    "CONSTITUTIONAL/VITAL SIGNS": "VITAL SIGNS",
    "BIOMETRIC DATA": "VITAL SIGNS",
    # REVIEW OF SYSTEMS
    "ROS": "REVIEW OF SYSTEMS",
    "REVIEW OF SYSTEM": "REVIEW OF SYSTEMS",
    "REVIEW OF SYMPTOMS": "REVIEW OF SYSTEMS",
    "SYSTEMS REVIEW": "REVIEW OF SYSTEMS",
    "REVIEW OF BODY SYSTEMS": "REVIEW OF SYSTEMS",
    "SIGNIFICANT ILLNESS AND REVIEW OF SYSTEMS": "REVIEW OF SYSTEMS",
    # CHIEF COMPLAINT
    "CHIEF COMPLAINTS": "CHIEF COMPLAINT",
    "CHIEF COMPLIANT": "CHIEF COMPLAINT",
    "CHIEF COMPLAINT / REASON FOR THE VISIT": "CHIEF COMPLAINT",
    "CHIEF COMPLAINT - REASON FOR VISIT": "CHIEF COMPLAINT",
    "CHIEF COMPLAINT/HISTORY OF PRESENT ILLNESS": "CHIEF COMPLAINT",
    "CHIEF COMPLAINT AND IDENTIFICATION": "CHIEF COMPLAINT",
    "REASON FOR VISIT": "CHIEF COMPLAINT",
    "REASON FOR ADMISSION": "CHIEF COMPLAINT",
    "REASON FOR ADMISSION/CHIEF COMPLAINT": "CHIEF COMPLAINT",
    "REASON FOR HOSPITALIZATION": "CHIEF COMPLAINT",
    "REASON FOR CONSULT": "REASON FOR CONSULTATION",
    "REASON FOR REFERRAL": "REASON FOR CONSULTATION",
    "REASON FOR EVALUATION": "REASON FOR CONSULTATION",
    "CHIEF REASON FOR CONSULTATION": "REASON FOR CONSULTATION",
    "INDICATION FOR CONSULTATION": "REASON FOR CONSULTATION",
    # ASSESSMENT AND PLAN
    "A/P": "ASSESSMENT AND PLAN",
    "ASSESSMENT & PLAN": "ASSESSMENT AND PLAN",
    "ASSESSMENT / PLAN": "ASSESSMENT AND PLAN",
    "ASSESSMENT/PLAN": "ASSESSMENT AND PLAN",
    "ASSESSMENT/PLAN/PROBLEMS": "ASSESSMENT AND PLAN",
    "IMPRESSION AND PLAN": "ASSESSMENT AND PLAN",
    "IMPRESSION/PLAN": "ASSESSMENT AND PLAN",
    "MPRESSION AND PLAN": "ASSESSMENT AND PLAN",
    "PLAN AND RECOMMENDATIONS": "ASSESSMENT AND PLAN",
    "PLAN/RECOMMENDATIONS": "ASSESSMENT AND PLAN",
    "PLAN / RECOMMENDATIONS": "ASSESSMENT AND PLAN",
    "DISCUSSION AND PLAN": "ASSESSMENT AND PLAN",
    "DISCUSSION/PLAN": "ASSESSMENT AND PLAN",
    "PLAN OF CARE": "ASSESSMENT AND PLAN",
    "PLAN/TREATMENT": "ASSESSMENT AND PLAN",
    "TREATMENT PLAN": "ASSESSMENT AND PLAN",
    "TREATMENT/PLAN": "ASSESSMENT AND PLAN",
    "EVALUATION/TREATMENT PLAN": "ASSESSMENT AND PLAN",
    "ASSESSMENT AND EVALUATION": "ASSESSMENT AND PLAN",
    "IMPRESSION DIAGNOSIS": "ASSESSMENT AND PLAN",
    "IMPRESSION / DIAGNOSIS": "ASSESSMENT AND PLAN",
    # IMPRESSION
    "FINAL IMPRESSION": "IMPRESSION",
    "OVERALL IMPRESSION": "IMPRESSION",
    "DIAGNOSTIC IMPRESSION": "IMPRESSION",
    "INITIAL IMPRESSION": "IMPRESSION",
    "CLINICAL IMPRESSION": "IMPRESSION",
    "IMPRESSIONS": "IMPRESSION",
    "IMPRESSION AT THIS TIME": "IMPRESSION",
    "IMPRESSION OF THE FINDINGS": "IMPRESSION",
    # ASSESSMENT
    "ASSESSMENTS": "ASSESSMENT",
    "ASSESSMENT RESULTS": "ASSESSMENT",
    # DIAGNOSES
    "DIAGNOSIS": "DIAGNOSES",
    "FINAL DIAGNOSIS": "DIAGNOSES",
    "FINAL DIAGNOSES": "DIAGNOSES",
    "CONCLUSIONS / FINAL DIAGNOSES": "DIAGNOSES",
    "PRIMARY DIAGNOSIS": "DIAGNOSES",
    "PRIMARY DIAGNOSES": "DIAGNOSES",
    "PRINCIPAL DIAGNOSIS": "DIAGNOSES",
    "PRINCIPAL DIAGNOSES": "DIAGNOSES",
    "CURRENT DIAGNOSES": "DIAGNOSES",
    "INITIAL DIAGNOSES": "DIAGNOSES",
    "PROBLEMS/DIAGNOSES": "DIAGNOSES",
    "PROBLEMS DIAGNOSES": "DIAGNOSES",
    "DIAGNOSES PROBLEMS": "DIAGNOSES",
    "DIFFERENTIAL DIAGNOSIS": "DIAGNOSES",
    "DIFFERENTIAL DIAGNOSES": "DIAGNOSES",
    # PREOPERATIVE DIAGNOSIS
    "PREOPERATIVE DIAGNOSES": "PREOPERATIVE DIAGNOSIS",
    "PREOP DIAGNOSIS": "PREOPERATIVE DIAGNOSIS",
    "PREOP DIAGNOSES": "PREOPERATIVE DIAGNOSIS",
    "PREOP DX": "PREOPERATIVE DIAGNOSIS",
    "PRE-OP DIAGNOSIS": "PREOPERATIVE DIAGNOSIS",
    "PRE-OP DIAGNOSES": "PREOPERATIVE DIAGNOSIS",
    "PRE-OPERATIVE DIAGNOSIS": "PREOPERATIVE DIAGNOSIS",
    "PRE-PROCEDURE DIAGNOSIS": "PREOPERATIVE DIAGNOSIS",
    "PREPROCEDURE DIAGNOSIS": "PREOPERATIVE DIAGNOSIS",
    "PREPROCEDURE DIAGNOSES": "PREOPERATIVE DIAGNOSIS",
    "PREOPERATIVE DX": "PREOPERATIVE DIAGNOSIS",
    "PREOPERATIVE DIAGNOSIS AND INDICATIONS": "PREOPERATIVE DIAGNOSIS",
    # POSTOPERATIVE DIAGNOSIS
    "POSTOPERATIVE DIAGNOSES": "POSTOPERATIVE DIAGNOSIS",
    "POSTOP DIAGNOSIS": "POSTOPERATIVE DIAGNOSIS",
    "POSTOP DIAGNOSES": "POSTOPERATIVE DIAGNOSIS",
    "POSTOP DX": "POSTOPERATIVE DIAGNOSIS",
    "POST-OP DIAGNOSIS": "POSTOPERATIVE DIAGNOSIS",
    "POST-OP DIAGNOSES": "POSTOPERATIVE DIAGNOSIS",
    "POST-OPERATIVE DIAGNOSIS": "POSTOPERATIVE DIAGNOSIS",
    "POST PROCEDURE DIAGNOSIS": "POSTOPERATIVE DIAGNOSIS",
    "POST-PROCEDURE DIAGNOSIS": "POSTOPERATIVE DIAGNOSIS",
    "POSTPROCEDURE DIAGNOSIS": "POSTOPERATIVE DIAGNOSIS",
    "POSTPROCEDURE DIAGNOSES": "POSTOPERATIVE DIAGNOSIS",
    "POSTOPERATIVE DX": "POSTOPERATIVE DIAGNOSIS",
    # PROCEDURE PERFORMED
    "PROCEDURES PERFORMED": "PROCEDURE PERFORMED",
    "PROCEDURES  PERFORMED": "PROCEDURE PERFORMED",
    "PROCEDURE  PERFORMED": "PROCEDURE PERFORMED",
    "OPERATIVE PROCEDURE": "PROCEDURE PERFORMED",
    "OPERATIVE PROCEDURES": "PROCEDURE PERFORMED",
    "OPERATION PERFORMED": "PROCEDURE PERFORMED",
    "OPERATIONS PERFORMED": "PROCEDURE PERFORMED",
    "SURGICAL PROCEDURE": "PROCEDURE PERFORMED",
    "SURGICAL PROCEDURE PERFORMED": "PROCEDURE PERFORMED",
    "OPERATIVE PROCEDURE PERFORMED": "PROCEDURE PERFORMED",
    "PRIMARY OPERATIVE PROCEDURE": "PROCEDURE PERFORMED",
    "PRINCIPAL PROCEDURE": "PROCEDURE PERFORMED",
    "NAME OF OPERATION": "PROCEDURE PERFORMED",
    "NAME OF PROCEDURE": "PROCEDURE PERFORMED",
    "TITLE OF OPERATION": "PROCEDURE PERFORMED",
    "TITLE OF PROCEDURE": "PROCEDURE PERFORMED",
    "TITLE OF PROCEDURES": "PROCEDURE PERFORMED",
    "TITLE OF THE OPERATION": "PROCEDURE PERFORMED",
    "TITLE OF THE PROCEDURE": "PROCEDURE PERFORMED",
    "TYPE OF OPERATION": "PROCEDURE PERFORMED",
    "TYPE OF PROCEDURE": "PROCEDURE PERFORMED",
    "OPERATION AND PROCEDURE": "PROCEDURE PERFORMED",
    "OPERATION PROCEDURE": "PROCEDURE PERFORMED",
    "OPERATION/PROCEDURE": "PROCEDURE PERFORMED",
    "OPERATIONS": "PROCEDURE PERFORMED",
    "OPERATIONS AND PROCEDURES": "PROCEDURE PERFORMED",
    "OPERATION": "PROCEDURE PERFORMED",
    # PROCEDURE
    "PROCEDURES": "PROCEDURE",
    "PROCEDURES DONE": "PROCEDURE",
    "PROCEDURE DONE": "PROCEDURE",
    "PROCEDURE NOTE": "PROCEDURE",
    "PROCEDURE REPORT": "PROCEDURE",
    "PROCEDURE CODES": "PROCEDURE",
    "PROCEDURE AND FINDINGS": "PROCEDURE",
    "PROCEDURE AND TECHNIQUE": "PROCEDURE",
    # DESCRIPTION OF PROCEDURE
    "DESCRIPTION OF OPERATION": "DESCRIPTION OF PROCEDURE",
    "DESCRIPTION OF OPERATIONS": "DESCRIPTION OF PROCEDURE",
    "DESCRIPTION OF OPERATIVE PROCEDURE": "DESCRIPTION OF PROCEDURE",
    "DESCRIPTION OF SURGERY": "DESCRIPTION OF PROCEDURE",
    "DESCRIPTION OF TECHNIQUE": "DESCRIPTION OF PROCEDURE",
    "DESCRIPTION OF THE CASE": "DESCRIPTION OF PROCEDURE",
    "DESCRIPTION OF THE OPERATION": "DESCRIPTION OF PROCEDURE",
    "DESCRIPTION OF THE PROCEDURE": "DESCRIPTION OF PROCEDURE",
    "DESCRIPTION OF THE PROCEDURE IN DETAIL": "DESCRIPTION OF PROCEDURE",
    "PROCEDURE DETAIL": "DESCRIPTION OF PROCEDURE",
    "PROCEDURE DETAILS": "DESCRIPTION OF PROCEDURE",
    "PROCEDURE IN DETAIL": "DESCRIPTION OF PROCEDURE",
    "PROCEDURE PERFORMED IN DETAIL": "DESCRIPTION OF PROCEDURE",
    "DETAIL OF THE OPERATION": "DESCRIPTION OF PROCEDURE",
    "DETAILED OPERATIVE NOTE": "DESCRIPTION OF PROCEDURE",
    "DETAILS": "DESCRIPTION OF PROCEDURE",
    "DETAILS OF PROCEDURE": "DESCRIPTION OF PROCEDURE",
    "DETAILS OF THE OPERATION": "DESCRIPTION OF PROCEDURE",
    "DETAILS OF THE PROCEDURE": "DESCRIPTION OF PROCEDURE",
    "OPERATIVE DETAILS": "DESCRIPTION OF PROCEDURE",
    "OPERATIVE NARRATIVE": "DESCRIPTION OF PROCEDURE",
    "OPERATIVE NOTE": "DESCRIPTION OF PROCEDURE",
    "OPERATIVE NOTE IN DETAIL": "DESCRIPTION OF PROCEDURE",
    "OPERATIVE REPORT": "DESCRIPTION OF PROCEDURE",
    "OPERATIVE REPORT IN DETAIL": "DESCRIPTION OF PROCEDURE",
    "OPERATIVE TECHNIQUE": "DESCRIPTION OF PROCEDURE",
    "OPERATION IN DETAIL": "DESCRIPTION OF PROCEDURE",
    "NARRATIVE OF PROCEDURE": "DESCRIPTION OF PROCEDURE",
    "PROCEDURE AND INTERPRETATION": "DESCRIPTION OF PROCEDURE",
    "PROCEDURE DESCRIPTION": "DESCRIPTION OF PROCEDURE",
    "TECHNICAL PROCEDURE": "DESCRIPTION OF PROCEDURE",
    # NEUROLOGICAL
    "NEUROLOGIC": "NEUROLOGICAL",
    "NEURO": "NEUROLOGICAL",
    "CNS": "NEUROLOGICAL",
    "CENTRAL NERVOUS SYSTEM": "NEUROLOGICAL",
    "NERVOUS SYSTEM": "NEUROLOGICAL",
    "GROSS NEUROLOGICAL EXAM": "NEUROLOGICAL",
    "OVERALL NEUROLOGICAL": "NEUROLOGICAL",
    "NEUROLOGIC EXAM": "NEUROLOGICAL",
    "NEUROLOGICAL EXAMINATION": "NEUROLOGICAL",
    # LABORATORY DATA
    "LAB": "LABORATORY DATA",
    "LABS": "LABORATORY DATA",
    "LAB DATA": "LABORATORY DATA",
    "LAB STUDIES": "LABORATORY DATA",
    "LAB TESTS": "LABORATORY DATA",
    "LABORATORY": "LABORATORY DATA",
    "LABORATORY RESULTS": "LABORATORY DATA",
    "LABORATORY STUDIES": "LABORATORY DATA",
    "LABORATORY TESTS": "LABORATORY DATA",
    "LABORATORY VALUES": "LABORATORY DATA",
    "LABORATORY FINDINGS": "LABORATORY DATA",
    "LABORATORY EXAMINATION": "LABORATORY DATA",
    "LABORATORY EXAM": "LABORATORY DATA",
    "LABORATORY AND DIAGNOSTIC DATA": "LABORATORY DATA",
    "LABORATORY AND X-RAY DATA": "LABORATORY DATA",
    "LABORATORY DATABASE": "LABORATORY DATA",
    "LABORATORY EVALUATION": "LABORATORY DATA",
    "PERTINENT LABORATORY DATA": "LABORATORY DATA",
    "PERTINENT LABORATORIES": "LABORATORY DATA",
    "LAB / TESTS": "LABORATORY DATA",
    "BLOOD STUDIES": "LABORATORY DATA",
    "DIAGNOSTIC DATA": "LABORATORY DATA",
    "DIAGNOSTIC STUDIES": "LABORATORY DATA",
    "DIAGNOSTIC AND LABORATORY DATA": "LABORATORY DATA",
    "DATA": "LABORATORY DATA",
    # HOSPITAL COURSE
    "COURSE": "HOSPITAL COURSE",
    "COURSE IN HOSPITAL": "HOSPITAL COURSE",
    "COURSE IN THE HOSPITAL": "HOSPITAL COURSE",
    "HOSPITAL COURSE AND TREATMENT": "HOSPITAL COURSE",
    "HOSPITAL COURSE PER PROBLEM LIST": "HOSPITAL COURSE",
    "HOSPITAL COURSE SUMMARY": "HOSPITAL COURSE",
    "HOSPITAL SUMMARY": "HOSPITAL COURSE",
    "BRIEF HOSPITAL COURSE": "HOSPITAL COURSE",
    "BRIEF HOSPITAL COURSE SUMMARY": "HOSPITAL COURSE",
    "BRIEF SUMMARY OF HOSPITAL COURSE": "HOSPITAL COURSE",
    # DISCHARGE DIAGNOSES
    "DISCHARGE DIAGNOSIS": "DISCHARGE DIAGNOSES",
    "DIAGNOSES ON DISCHARGE": "DISCHARGE DIAGNOSES",
    "CONDITION AT DISCHARGE": "DISCHARGE DIAGNOSES",
    "ADDITIONAL DISCHARGE DIAGNOSES": "DISCHARGE DIAGNOSES",
    "PRIMARY DISCHARGE DIAGNOSES": "DISCHARGE DIAGNOSES",
    "SECONDARY DISCHARGE DIAGNOSES": "DISCHARGE DIAGNOSES",
    # INDICATIONS
    "INDICATION": "INDICATIONS",
    "INDICATIONS FOR OPERATION": "INDICATIONS",
    "INDICATIONS FOR PROCEDURE": "INDICATIONS",
    "INDICATIONS FOR PROCEDURES": "INDICATIONS",
    "INDICATIONS FOR SURGERY": "INDICATIONS",
    "INDICATIONS FOR THE OPERATION": "INDICATIONS",
    "INDICATIONS FOR THE PROCEDURE": "INDICATIONS",
    "INDICATIONS FOR THE SURGERY": "INDICATIONS",
    "INDICATIONS FOR THIS PROCEDURE": "INDICATIONS",
    "INDICATIONS OF OPERATION": "INDICATIONS",
    "INDICATIONS OF PROCEDURE": "INDICATIONS",
    "INDICATIONS OF SURGERY": "INDICATIONS",
    "INDICATIONS OF THE PROCEDURE": "INDICATIONS",
    "INDICATIONS AND CONCERNS": "INDICATIONS",
    "INDICATION FOR OPERATION": "INDICATIONS",
    "INDICATION FOR PROCEDURE": "INDICATIONS",
    "INDICATION FOR STUDY": "INDICATIONS",
    "INDICATION FOR SURGERY": "INDICATIONS",
    "INDICATION FOR THE OPERATION": "INDICATIONS",
    "INDICATION FOR THE PROCEDURE": "INDICATIONS",
    "SURGICAL INDICATIONS": "INDICATIONS",
    "OPERATIVE INDICATIONS": "INDICATIONS",
    "PROCEDURE INDICATION": "INDICATIONS",
    "REASON FOR PROCEDURE": "INDICATIONS",
    "REASON FOR EXAM": "INDICATIONS",
    "REASON FOR EXAMINATION": "INDICATIONS",
    "REASON FOR CT SCAN": "INDICATIONS",
    "REASON FOR THE TEST": "INDICATIONS",
    "CLINICAL INDICATION": "INDICATIONS",
    "CLINICAL INDICATIONS": "INDICATIONS",
    "CLINICAL INFORMATION": "INDICATIONS",
    # GENERAL
    "GENERAL APPEARANCE": "GENERAL",
    "GEN": "GENERAL",
    "GEN EXAM": "GENERAL",
    "GENERAL OBSERVATIONS": "GENERAL",
    "GENERAL EVALUATION": "GENERAL",
    "GENERAL/CONSTITUTIONAL": "GENERAL",
    "CONSTITUTIONAL": "GENERAL",
    "CONSTITUTIONAL SYMPTOMS": "GENERAL",
    # FINDINGS
    "FINDING": "FINDINGS",
    "INTRAOPERATIVE FINDINGS": "FINDINGS",
    "INTRAOPERATIVE FINDING": "FINDINGS",
    "OPERATIVE FINDINGS": "FINDINGS",
    "GROSS FINDINGS": "FINDINGS",
    "DESCRIPTION OF FINDINGS": "FINDINGS",
    "FINDINGS AT OPERATION": "FINDINGS",
    "FINDINGS AT THE TIME OF SURGERY": "FINDINGS",
    "FINDINGS AT TIME OF SURGERY": "FINDINGS",
    "FINDINGS DURING THE PROCEDURE": "FINDINGS",
    "FINDINGS OF THE OPERATION": "FINDINGS",
    "ANGIOGRAPHIC FINDINGS": "FINDINGS",
    "ENDOSCOPIC FINDINGS": "FINDINGS",
    "GROSS INTRAOPERATIVE FINDINGS": "FINDINGS",
    "GROSS OPERATIVE FINDINGS": "FINDINGS",
    "POSTOPERATIVE FINDINGS": "FINDINGS",
    "PREOPERATIVE FINDINGS": "FINDINGS",
    "PATHOLOGICAL FINDINGS": "FINDINGS",
    "MAJOR FINDINGS": "FINDINGS",
    "PERTINENT FINDINGS": "FINDINGS",
    "SUMMARY OF FINDINGS": "FINDINGS",
    # HEENT
    "HEENT AND NECK": "HEENT",
    "HEENT/NECK": "HEENT",
    "ENMT": "HEENT",
    "ENT": "HEENT",
    "HEAD AND NECK": "HEENT",
    "HEAD AND NECK EXAMINATION": "HEENT",
    # CARDIOVASCULAR
    "CARDIOVASCULAR SYSTEM": "CARDIOVASCULAR",
    "CARDIOVASCULAR EXAM": "CARDIOVASCULAR",
    "CARDIAC": "CARDIOVASCULAR",
    "COR": "CARDIOVASCULAR",
    "CVS": "CARDIOVASCULAR",
    "CV - RESP": "CARDIOVASCULAR",
    "HEART": "CARDIOVASCULAR",
    # RESPIRATORY
    "RESPIRATORY SYSTEM": "RESPIRATORY",
    "PULM": "RESPIRATORY",
    "PULMONARY": "RESPIRATORY",
    "LUNGS": "RESPIRATORY",
    "CHEST": "RESPIRATORY",
    # MUSCULOSKELETAL
    "MUSCULOSKELETAL SYSTEM": "MUSCULOSKELETAL",
    "MUSCULOSKELETAL/EXTREMITIES": "MUSCULOSKELETAL",
    "SKELETAL": "MUSCULOSKELETAL",
    "SKELETAL SYSTEM": "MUSCULOSKELETAL",
    "JOINTS": "MUSCULOSKELETAL",
    # GENITOURINARY
    "GENITOURINARY SYSTEM": "GENITOURINARY",
    "GU EXAM": "GENITOURINARY",
    "GU/RECTAL": "GENITOURINARY",
    "UROLOGIC": "GENITOURINARY",
    "UROLOGICAL": "GENITOURINARY",
    # GASTROINTESTINAL
    "GASTROINTESTINAL SYSTEM": "GASTROINTESTINAL",
    "ABDOMEN": "GASTROINTESTINAL",
    "ABD": "GASTROINTESTINAL",
    # PSYCHIATRIC
    "PSYCHE": "PSYCHIATRIC",
    "PSYCH": "PSYCHIATRIC",
    "PSYCHOLOGICAL": "PSYCHIATRIC",
    "PSYCHOLOGIC": "PSYCHIATRIC",
    "MENTAL STATUS": "PSYCHIATRIC",
    "MENTAL STATUS EXAM": "PSYCHIATRIC",
    "MENTAL STATUS EXAMINATION": "PSYCHIATRIC",
    "MENTAL STATUS EVALUATION": "PSYCHIATRIC",
    "MSE": "PSYCHIATRIC",
    # SKIN
    "DERMATOLOGIC": "SKIN",
    "DERM": "SKIN",
    "INTEGUMENTARY": "SKIN",
    # HEMATOLOGY
    "HEMATOLOGIC": "HEMATOLOGY",
    "HEMATOLOGICAL": "HEMATOLOGY",
    "HEME/LYMPH": "HEMATOLOGY",
    "HEMATOPOIETIC": "HEMATOLOGY",
    "HEMATOLOGIC AND LYMPHATIC": "HEMATOLOGY",
    "HEMATOLOGIC/LYMPHATIC": "HEMATOLOGY",
    # RECOMMENDATIONS
    "RECOMMENDATION": "RECOMMENDATIONS",
    "PLAN AND SUGGESTION": "RECOMMENDATIONS",
    "PLANS/RECOMMENDATIONS": "RECOMMENDATIONS",
    "PLANS": "RECOMMENDATIONS",
    "ASSESSMENT AND RECOMMENDATIONS": "RECOMMENDATIONS",
    "EVALUATION/RECOMMENDATION": "RECOMMENDATIONS",
    # SUMMARY
    "SUMMARY AND CONCLUSIONS": "SUMMARY",
    "SUMMARY OF CLINICAL HISTORY": "SUMMARY",
    "CLINICAL SUMMARY": "SUMMARY",
    "CLINICAL RESUME": "SUMMARY",
    "IN SUMMARY": "SUMMARY",
    "ADMISSION SUMMARY": "SUMMARY",
    # ANESTHESIA
    "ANESTHETIC": "ANESTHESIA",
    "ANESTHESIOLOGY": "ANESTHESIA",
    "ANESTHESIOLOGIST": "ANESTHESIA",
    "ANESTHESIA/SEDATION": "ANESTHESIA",
    "TYPE OF ANESTHESIA": "ANESTHESIA",
    "SECOND ANESTHESIA": "ANESTHESIA",
    "THIRD ANESTHESIA": "ANESTHESIA",
    "SEDATION": "ANESTHESIA",
    "LOCAL ANESTHETIC": "ANESTHESIA",
    "LOCAL": "ANESTHESIA",
    "IV SEDATION": "ANESTHESIA",
    # ESTIMATED BLOOD LOSS
    "EBL": "ESTIMATED BLOOD LOSS",
    "BLOOD LOSS": "ESTIMATED BLOOD LOSS",
    # FLUIDS
    "FLUID": "FLUIDS",
    "IV FLUIDS": "FLUIDS",
    "INTRAVENOUS FLUIDS": "FLUIDS",
    "INTRAOPERATIVE FLUIDS": "FLUIDS",
    "FLUIDS GIVEN": "FLUIDS",
    "FLUIDS RECEIVED": "FLUIDS",
    "FLUID RECEIVED": "FLUIDS",
    "FLUID REPLACEMENT": "FLUIDS",
    "CRYSTALLOIDS": "FLUIDS",
    # DISPOSITION
    "DISCHARGE DISPOSITION": "DISPOSITION",
    "CONDITION ON DISPOSITION": "DISPOSITION",
    "CONDITION UPON DISPOSITION": "DISPOSITION",
    "AFTERCARE AND DISPOSITION": "DISPOSITION",
    "POST-PROCEDURE COURSE AND DISPOSITION": "DISPOSITION",
    # RADIOLOGY
    "RADIOLOGIC DATA": "RADIOLOGY",
    "RADIOLOGICAL DATA": "RADIOLOGY",
    "RADIOGRAPHIC DATA": "RADIOLOGY",
    "RADIOGRAPHIC STUDIES": "RADIOLOGY",
    "RADIOGRAPHS": "RADIOLOGY",
    "IMAGING": "RADIOLOGY",
    "IMAGING STUDIES": "RADIOLOGY",
    "IMAGING DATA": "RADIOLOGY",
    "DIAGNOSTIC IMAGING": "RADIOLOGY",
    "MEDICAL IMAGING": "RADIOLOGY",
    "X-RAYS": "RADIOLOGY",
    "X-RAY AND LABORATORY DATA": "RADIOLOGY",
    "X-RAY INTERPRETATION": "RADIOLOGY",
    # TECHNIQUE
    "IMAGE TECHNIQUE": "TECHNIQUE",
    "PROCEDURE TECHNIQUE": "TECHNIQUE",
    "STUDY PROTOCOL": "TECHNIQUE",
    "NUCLEAR PROTOCOL": "TECHNIQUE",
    "STRESS TECHNIQUE": "TECHNIQUE",
    # SUBJECTIVE
    "SUBJECTIVE COMPLAINTS": "SUBJECTIVE",
    # FOLLOW-UP
    "FOLLOW UP": "FOLLOW-UP",
    "FOLLOWUP": "FOLLOW-UP",
    "FOLLOW-UP APPOINTMENT": "FOLLOW-UP",
    "FOLLOWUP APPOINTMENTS": "FOLLOW-UP",
    "FOLLOWUP CARE": "FOLLOW-UP",
    "DISCHARGE FOLLOWUP": "FOLLOW-UP",
    "DISCHARGE FOLLOWUP PLANNING": "FOLLOW-UP",
    "OUTPATIENT CARE": "FOLLOW-UP",
    # SPECIMENS
    "SPECIMEN": "SPECIMENS",
    "PATHOLOGY": "SPECIMENS",
    "PATHOLOGY SPECIMEN": "SPECIMENS",
    "GROSS DESCRIPTION": "SPECIMENS",
    "MICROSCOPIC DESCRIPTION": "SPECIMENS",
    "MICROSCOPIC EXAMINATION": "SPECIMENS",
}


def normalize_section(section: str) -> str:
    if not section:
        return section
    return SECTION_MAP.get(section.strip().upper(), section.strip().upper())


# ── embedding ─────────────────────────────────────────────────────────────────

def embed_batch(model: str, texts: list[str]) -> list[list[float]]:
    response = ollama.embed(model=model, input=texts)
    return response["embeddings"]


def safe_embed_and_upsert(collection, batch: list[dict], model: str):
    texts = [c["text"] for c in batch]
    try:
        embeddings = embed_batch(model, texts)
    except Exception as e:
        print(f"\n  Batch failed ({e}), 1-by-1 fallback...")
        embeddings = []
        for t in texts:
            try:
                embeddings.extend(embed_batch(model, [t]))
            except Exception:
                embeddings.append(None)
        valid = [(c, e) for c, e in zip(batch, embeddings) if e is not None]
        if not valid:
            return
        batch_list, emb_list = zip(*valid)
        batch, embeddings, texts = list(batch_list), list(emb_list), [c["text"] for c in batch_list]

    collection.upsert(
        ids=[c["id"] for c in batch],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c["metadata"] for c in batch],
    )


# ── step 1: normalize SQLite ──────────────────────────────────────────────────

def step_normalize_sqlite():
    print("\n[1/4] Normalizing section names in SQLite...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT section FROM chunks")
    sections = [r[0] for r in cur.fetchall()]
    updated = 0
    for section in sections:
        canonical = normalize_section(section)
        if canonical != section:
            cur.execute("UPDATE chunks SET section = ? WHERE section = ?", (canonical, section))
            updated += cur.rowcount
    conn.commit()
    cur.execute("SELECT COUNT(DISTINCT section) FROM chunks")
    after = cur.fetchone()[0]
    conn.close()
    print(f"  Done. {updated:,} rows updated. {after} distinct sections remaining.")


# ── step 2: embed mtsamples ───────────────────────────────────────────────────

def step_embed_mtsamples(model: str, batch_size: int, collection):
    print("\n[2/4] Embedding mtsamples chunks...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "SELECT chunk_id, text, specialty, section, keywords, source, row_index FROM chunks"
    )
    existing = set(collection.get(include=[])["ids"])
    pending = []
    for row in cur.fetchall():
        chunk_id, text, specialty, section, keywords_raw, source, row_index = row
        if chunk_id in existing:
            continue
        try:
            kw_list = json.loads(keywords_raw) if keywords_raw else []
            keywords_str = ", ".join(str(k) for k in kw_list)
        except Exception:
            keywords_str = str(keywords_raw or "")
        pending.append({
            "id": chunk_id,
            "text": clean(text or ""),
            "metadata": {
                "specialty": specialty or "",
                "section": normalize_section(section or ""),
                "keywords": keywords_str,
                "source": source or "mtsamples",
                "row_index": int(row_index or 0),
            },
        })
    conn.close()
    print(f"  {len(existing):,} already embedded. {len(pending):,} to go.")
    with tqdm(total=len(pending), desc="  mtsamples_chunks", unit="chunk") as bar:
        for i in range(0, len(pending), batch_size):
            batch = pending[i: i + batch_size]
            safe_embed_and_upsert(collection, batch, model)
            bar.update(len(batch))
    print(f"  mtsamples_chunks total: {collection.count():,}")


# ── step 3: embed patients ────────────────────────────────────────────────────

def get_patient_name(patient: dict) -> str:
    names = patient.get("name", [])
    if not names:
        return "Unknown"
    n = names[0]
    return f"{'  '.join(n.get('given', []))} {n.get('family', '')}".strip()


def calc_age(dob: str, deceased: str | None) -> int:
    if not dob:
        return 0
    from datetime import date
    birth = date.fromisoformat(dob)
    end = date.fromisoformat(deceased[:10]) if deceased else date.today()
    return (end - birth).days // 365


def decode_docref(resource: dict) -> str:
    try:
        b64 = resource["content"][0]["attachment"].get("data", "")
        return base64.b64decode(b64).decode("utf-8") if b64 else ""
    except Exception:
        return ""


def extract_fhir(fhir_file: Path) -> tuple[list[dict], dict | None]:
    try:
        data = json.loads(fhir_file.read_text())
    except Exception:
        return [], None

    by_type: dict[str, list] = {}
    for entry in data.get("entry", []):
        res = entry.get("resource", {})
        by_type.setdefault(res.get("resourceType", ""), []).append(res)

    patients = by_type.get("Patient", [])
    if not patients:
        return [], None
    patient = patients[0]

    pid = patient["id"]
    name = get_patient_name(patient)
    dob = patient.get("birthDate", "")
    gender = patient.get("gender", "")
    deceased = patient.get("deceasedDateTime")
    age = calc_age(dob, deceased)

    chunks = []

    # demographics
    chunks.append({
        "id": f"{pid}__demographics",
        "text": clean(f"Patient: {name}. DOB: {dob}. Gender: {gender}. Age: {age}. Status: {'deceased' if deceased else 'alive'}."),
        "metadata": {"patient_id": pid, "patient_name": name, "chunk_type": "demographics", "source": "fhir", "date": dob},
    })

    # conditions
    conditions = by_type.get("Condition", [])
    if conditions:
        active, resolved = [], []
        for c in conditions:
            display = c.get("code", {}).get("coding", [{}])[0].get("display", "") or c.get("code", {}).get("text", "")
            status = c.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "")
            onset = c.get("onsetDateTime", "")[:10]
            (active if status == "active" else resolved).append(f"{display} (onset: {onset})")
        text = f"Patient {name} conditions. "
        if active:
            text += "Active: " + "; ".join(active) + ". "
        if resolved:
            text += "Resolved: " + "; ".join(resolved[:10]) + "."
        chunks.append({"id": f"{pid}__conditions", "text": clean(text),
                       "metadata": {"patient_id": pid, "patient_name": name, "chunk_type": "conditions", "source": "fhir", "date": dob}})

    # medications
    meds = by_type.get("MedicationRequest", [])
    if meds:
        active_m, past_m = [], []
        for m in meds:
            display = m.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display", "") or m.get("medicationCodeableConcept", {}).get("text", "")
            (active_m if m.get("status") == "active" else past_m).append(display)
        text = f"Patient {name} medications. "
        if active_m:
            text += "Current: " + "; ".join(active_m) + ". "
        if past_m:
            text += "Past: " + "; ".join(past_m[:10]) + "."
        chunks.append({"id": f"{pid}__medications", "text": clean(text),
                       "metadata": {"patient_id": pid, "patient_name": name, "chunk_type": "medications", "source": "fhir", "date": dob}})

    # encounters
    encounters = by_type.get("Encounter", [])
    if encounters:
        lines = []
        for enc in encounters[-20:]:
            etype = enc.get("type", [{}])[0].get("coding", [{}])[0].get("display", "") or enc.get("type", [{}])[0].get("text", "")
            period = enc.get("period", {}).get("start", "")[:10]
            rc = enc.get("reasonCode", [])
            reason = rc[0].get("coding", [{}])[0].get("display", "") if rc else ""
            lines.append(f"{period}: {etype}" + (f" for {reason}" if reason else ""))
        last_date = max((e.get("period", {}).get("start", "") for e in encounters), default="")[:10]
        chunks.append({"id": f"{pid}__encounters", "text": clean(f"Patient {name} encounters. " + ". ".join(lines) + "."),
                       "metadata": {"patient_id": pid, "patient_name": name, "chunk_type": "encounters", "source": "fhir", "date": last_date}})

    # document references
    for i, doc in enumerate(by_type.get("DocumentReference", [])):
        text = decode_docref(doc)
        if not text.strip():
            continue
        chunks.append({"id": f"{pid}__docref_{i}", "text": clean(text),
                       "metadata": {"patient_id": pid, "patient_name": name, "chunk_type": "clinical_note", "source": "fhir_docref", "date": doc.get("date", "")[:10]}})

    # patient_index summary
    active_cond_names = [
        c.get("code", {}).get("coding", [{}])[0].get("display", "") or c.get("code", {}).get("text", "")
        for c in conditions
        if c.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "") == "active"
    ]
    active_med_names = [
        m.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display", "") or m.get("medicationCodeableConcept", {}).get("text", "")
        for m in meds if m.get("status") == "active"
    ]
    last_enc = max((e.get("period", {}).get("start", "") for e in encounters), default="")[:10] if encounters else ""
    summary = {
        "id": pid,
        "text": clean(
            f"{name}, {age} year old {gender}. DOB {dob}. {'Deceased.' if deceased else ''} "
            f"Active conditions: {', '.join(active_cond_names) or 'none'}. "
            f"Current medications: {', '.join(active_med_names) or 'none'}. "
            f"Last encounter: {last_enc or 'unknown'}. "
            f"Total encounters: {len(encounters)}. Total conditions: {len(conditions)}."
        ),
        "metadata": {
            "patient_id": pid, "patient_name": name, "dob": dob, "gender": gender,
            "age": age, "deceased": deceased or "", "condition_count": len(conditions),
            "active_condition_count": len(active_cond_names), "med_count": len(meds),
            "active_med_count": len(active_med_names), "encounter_count": len(encounters),
            "last_encounter": last_enc, "source": "fhir",
        },
    }
    return chunks, summary


def step_embed_patients(model: str, batch_size: int, col_structured, col_index):
    print("\n[3/4] Embedding FHIR patients + clinician notes...")
    fhir_files = sorted(FHIR_PATH.glob("*.json"))
    print(f"  {len(fhir_files)} FHIR files found")

    existing_structured = set(col_structured.get(include=[])["ids"])
    existing_index = set(col_index.get(include=[])["ids"])

    all_structured, all_index = [], []

    for fhir_file in tqdm(fhir_files, desc="  Parsing FHIR", unit="patient"):
        chunks, summary = extract_fhir(fhir_file)

        for c in chunks:
            if c["id"] not in existing_structured:
                all_structured.append(c)

        if summary and summary["id"] not in existing_index:
            pid = summary["metadata"]["patient_id"]
            notes_file = NOTES_PATH / f"{pid}.json"
            if notes_file.exists():
                try:
                    notes = json.loads(notes_file.read_text())
                    for j, note in enumerate(notes):
                        text = note.get("note_text", "").strip()
                        if not text:
                            continue
                        nid = f"{pid}__clinician_note_{j}"
                        if nid not in existing_structured:
                            all_structured.append({
                                "id": nid,
                                "text": clean(text),
                                "metadata": {
                                    "patient_id": pid,
                                    "patient_name": note.get("patient_name", summary["metadata"]["patient_name"]),
                                    "chunk_type": "clinician_note",
                                    "note_type": note.get("note_type", ""),
                                    "source": "clinician_note",
                                    "date": note.get("timestamp", "")[:10],
                                },
                            })
                except Exception:
                    pass
            all_index.append(summary)

    print(f"  synthea_structured: {len(existing_structured):,} existing + {len(all_structured):,} new")
    print(f"  patient_index:      {len(existing_index):,} existing + {len(all_index):,} new")

    with tqdm(total=len(all_structured), desc="  synthea_structured", unit="chunk") as bar:
        for i in range(0, len(all_structured), batch_size):
            batch = all_structured[i: i + batch_size]
            safe_embed_and_upsert(col_structured, batch, model)
            bar.update(len(batch))

    with tqdm(total=len(all_index), desc="  patient_index", unit="patient") as bar:
        for i in range(0, len(all_index), batch_size):
            batch = all_index[i: i + batch_size]
            safe_embed_and_upsert(col_index, batch, model)
            bar.update(len(batch))

    print(f"  synthea_structured total: {col_structured.count():,}")
    print(f"  patient_index total:      {col_index.count():,}")


# ── step 4: setup logs ────────────────────────────────────────────────────────

def step_setup_logs():
    print("\n[4/4] Setting up logs DB...")
    conn = sqlite3.connect(LOGS_DB)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            query_text  TEXT NOT NULL,
            collection  TEXT NOT NULL,
            filters     TEXT,
            n_results   INTEGER,
            latency_ms  INTEGER
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            patient_id  TEXT,
            patient_name TEXT,
            action      TEXT NOT NULL,
            query_text  TEXT,
            collection  TEXT,
            source      TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_activity_timestamp ON activity_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_log(patient_id);
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
    """)
    conn.commit()
    conn.close()
    print(f"  Logs DB ready: {LOGS_DB}")


# ── entrypoint ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-embedding:8b")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--reset", action="store_true", help="Drop all collections and re-embed from scratch")
    p.add_argument("--only", choices=["mtsamples", "patients", "logs"], help="Run only one step")
    return p.parse_args()


def main():
    args = parse_args()

    if args.only != "logs":
        try:
            models = [m.model for m in ollama.list().models]
        except Exception as e:
            print(f"ERROR: Cannot reach Ollama — {e}\nStart with: ollama serve")
            sys.exit(1)
        if args.model not in models:
            print(f"ERROR: Model '{args.model}' not found.\nPull: ollama pull {args.model}")
            sys.exit(1)

    CHROMA_PATH.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if args.reset:
        for name in ["mtsamples_chunks", "synthea_structured", "patient_index"]:
            try:
                client.delete_collection(name)
                print(f"Dropped '{name}'")
            except Exception:
                pass

    col_mtsamples = client.get_or_create_collection("mtsamples_chunks", metadata={"hnsw:space": "cosine"})
    col_structured = client.get_or_create_collection("synthea_structured", metadata={"hnsw:space": "cosine"})
    col_index = client.get_or_create_collection("patient_index", metadata={"hnsw:space": "cosine"})

    only = args.only

    if only is None or only == "mtsamples":
        step_normalize_sqlite()
        step_embed_mtsamples(args.model, args.batch_size, col_mtsamples)

    if only is None or only == "patients":
        step_embed_patients(args.model, args.batch_size, col_structured, col_index)

    if only is None or only == "logs":
        step_setup_logs()

    print("\nAll done.")
    print(f"  mtsamples_chunks:    {col_mtsamples.count():,}")
    print(f"  synthea_structured:  {col_structured.count():,}")
    print(f"  patient_index:       {col_index.count():,}")


if __name__ == "__main__":
    main()
