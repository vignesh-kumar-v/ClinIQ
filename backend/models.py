from pydantic import BaseModel
from typing import Optional


class QueryRequest(BaseModel):
    message: str
    session_id: str


class QueryResponse(BaseModel):
    answer: str
    intent: str  # read | write | mixed
    sources: list[str]


class NoteRequest(BaseModel):
    text: str
    note_type: str  # medication_review | lab_review | annual_physical | follow_up


class MedicationUpdateRequest(BaseModel):
    drug: str
    field: str  # dose | frequency
    value: str


class MedicationInput(BaseModel):
    drug: str
    dose: str
    frequency: str


class NewPatientRequest(BaseModel):
    name: str
    age: int
    gender: str
    conditions: list[str] = []
    medications: list[MedicationInput] = []
    notes: str = ""


class NewPatientResponse(BaseModel):
    patient_id: str
    status: str


class IngestSyntheaRequest(BaseModel):
    file_path: str


class IngestMTSamplesRequest(BaseModel):
    text: str
    specialty: str


class ActivityEntry(BaseModel):
    action: str
    detail: str
    timestamp: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | processing | complete | failed
    progress: int
    total: int
    message: str
