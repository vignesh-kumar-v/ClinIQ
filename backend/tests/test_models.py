import pytest
from models import (
    QueryRequest, NoteRequest, MedicationUpdateRequest,
    MedicationInput, NewPatientRequest, RegisterRequest, LoginRequest,
    IngestSyntheaRequest, IngestMTSamplesRequest,
)


class TestQueryRequest:
    def test_valid(self):
        q = QueryRequest(message="Hello", session_id="test-123")
        assert q.message == "Hello"
        assert q.session_id == "test-123"

    def test_empty_message(self):
        with pytest.raises(ValueError):
            QueryRequest(message="", session_id="test-123")

    def test_message_too_long(self):
        with pytest.raises(ValueError):
            QueryRequest(message="x" * 4001, session_id="test-123")

    def test_invalid_session_id(self):
        with pytest.raises(ValueError):
            QueryRequest(message="Hello", session_id="bad session!")


class TestNoteRequest:
    def test_valid(self):
        n = NoteRequest(text="Patient stable", note_type="follow_up")
        assert n.text == "Patient stable"

    def test_invalid_note_type(self):
        with pytest.raises(ValueError):
            NoteRequest(text="ok", note_type="invalid_type")


class TestMedicationUpdateRequest:
    def test_valid(self):
        m = MedicationUpdateRequest(drug="Aspirin", field="dose", value="100mg")
        assert m.drug == "Aspirin"

    def test_invalid_field(self):
        with pytest.raises(ValueError):
            MedicationUpdateRequest(drug="Aspirin", field="color", value="red")


class TestNewPatientRequest:
    def test_valid(self):
        p = NewPatientRequest(name="John", age=30, gender="male")
        assert p.name == "John"

    def test_invalid_age(self):
        with pytest.raises(ValueError):
            NewPatientRequest(name="John", age=200, gender="male")

    def test_invalid_gender(self):
        with pytest.raises(ValueError):
            NewPatientRequest(name="John", age=30, gender="unknown")

    def test_condition_too_long(self):
        with pytest.raises(ValueError):
            NewPatientRequest(name="John", age=30, gender="male", conditions=["x" * 501])


class TestRegisterRequest:
    def test_valid(self):
        r = RegisterRequest(email="a@b.com", password="123456", name="Test")
        assert r.email == "a@b.com"

    def test_invalid_email(self):
        with pytest.raises(ValueError):
            RegisterRequest(email="notanemail", password="123456")

    def test_short_password(self):
        with pytest.raises(ValueError):
            RegisterRequest(email="a@b.com", password="12345")


class TestLoginRequest:
    def test_valid(self):
        l = LoginRequest(email="a@b.com", password="123456")
        assert l.email == "a@b.com"


class TestIngestSyntheaRequest:
    def test_valid(self):
        r = IngestSyntheaRequest(file_path="/data/patient.json")
        assert r.file_path == "/data/patient.json"

    def test_empty_path(self):
        with pytest.raises(ValueError):
            IngestSyntheaRequest(file_path="")


class TestIngestMTSamplesRequest:
    def test_valid(self):
        r = IngestMTSamplesRequest(text="Sample text", specialty="cardiology")
        assert r.specialty == "cardiology"

    def test_empty_text(self):
        with pytest.raises(ValueError):
            IngestMTSamplesRequest(text="", specialty="cardiology")
