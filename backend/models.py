from pydantic import BaseModel, Field, field_validator
from typing import Literal


class QueryRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")


class QueryResponse(BaseModel):
    answer: str
    intent: str
    sources: list[str]


class NoteRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    note_type: Literal["medication_review", "lab_review", "annual_physical", "follow_up"]


class MedicationUpdateRequest(BaseModel):
    drug: str = Field(min_length=1, max_length=200)
    field: Literal["dose", "frequency"]
    value: str = Field(min_length=1, max_length=500)


class MedicationInput(BaseModel):
    drug: str = Field(min_length=1, max_length=200)
    dose: str = Field(min_length=1, max_length=200)
    frequency: str = Field(min_length=1, max_length=200)


class NewPatientRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    age: int = Field(ge=0, le=150)
    gender: Literal["male", "female", "other"]
    conditions: list[str] = Field(default_factory=list, max_length=50)
    medications: list[MedicationInput] = Field(default_factory=list, max_length=50)
    notes: str = Field(default="", max_length=10000)

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, v: list[str]) -> list[str]:
        for item in v:
            if len(item) > 500:
                raise ValueError(f"Condition too long: {item[:50]}...")
        return v


class NewPatientResponse(BaseModel):
    patient_id: str
    status: str


class IngestSyntheaRequest(BaseModel):
    file_path: str = Field(min_length=1, max_length=500)


class IngestMTSamplesRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)
    specialty: str = Field(min_length=1, max_length=200)


class ActivityEntry(BaseModel):
    action: str
    detail: str
    timestamp: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    total: int
    message: str
