import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestAuthEndpoints:
    def test_register_success(self):
        import uuid
        email = f"api-test-{uuid.uuid4().hex[:8]}@cliniq.com"
        resp = client.post("/api/auth/register", json={
            "email": email,
            "password": "test1234",
            "name": "API Tester",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["email"] == email
        assert data["token_type"] == "bearer"

    def test_register_duplicate(self):
        import uuid
        email = f"dup-{uuid.uuid4().hex[:8]}@cliniq.com"
        client.post("/api/auth/register", json={
            "email": email, "password": "test1234",
        })
        resp = client.post("/api/auth/register", json={
            "email": email, "password": "test1234",
        })
        assert resp.status_code == 409

    def test_register_invalid_email(self):
        resp = client.post("/api/auth/register", json={
            "email": "notanemail", "password": "test1234",
        })
        assert resp.status_code == 422

    def test_register_short_password(self):
        resp = client.post("/api/auth/register", json={
            "email": "short@pw.com", "password": "12345",
        })
        assert resp.status_code == 422

    def test_login_success(self):
        import uuid
        email = f"login-{uuid.uuid4().hex[:8]}@cliniq.com"
        client.post("/api/auth/register", json={
            "email": email, "password": "test1234",
        })
        resp = client.post("/api/auth/login", json={
            "email": email, "password": "test1234",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data

    def test_login_wrong_password(self):
        import uuid
        email = f"wrongpw-{uuid.uuid4().hex[:8]}@cliniq.com"
        client.post("/api/auth/register", json={
            "email": email, "password": "correct",
        })
        resp = client.post("/api/auth/login", json={
            "email": email, "password": "wrong",
        })
        assert resp.status_code == 401

    def test_login_nonexistent(self):
        resp = client.post("/api/auth/login", json={
            "email": "nobody@cliniq.com", "password": "test1234",
        })
        assert resp.status_code == 401


class TestProtectedEndpoints:
    @pytest.fixture
    def token(self):
        import uuid
        email = f"test-{uuid.uuid4().hex[:8]}@cliniq.com"
        resp = client.post("/api/auth/register", json={
            "email": email, "password": "test1234",
        })
        return resp.json()["access_token"]

    def _headers(self, token):
        return {"Authorization": f"Bearer {token}"}

    def test_list_patients_authenticated(self, token):
        resp = client.get("/api/patients?limit=1", headers=self._headers(token))
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "patients" in data

    def test_list_patients_unauthenticated(self):
        resp = client.get("/api/patients?limit=1")
        assert resp.status_code == 401

    def test_list_patients_invalid_token(self):
        resp = client.get("/api/patients?limit=1", headers={"Authorization": "Bearer badtoken"})
        assert resp.status_code == 401

    def test_get_patient_authenticated(self, token):
        list_resp = client.get("/api/patients?limit=1", headers=self._headers(token))
        pid = list_resp.json()["patients"][0]["patient_id"]
        resp = client.get(f"/api/patients/{pid}", headers=self._headers(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["patient_id"] == pid

    def test_get_patient_unauthenticated(self):
        resp = client.get("/api/patients/some-id")
        assert resp.status_code == 401

    def test_query_authenticated(self, token):
        resp = client.post("/api/patients/test/query", json={
            "message": "What are the symptoms of diabetes?",
            "session_id": "api-test-session",
        }, headers=self._headers(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] in ("read", "write", "mixed", "out_of_scope")
        assert "answer" in data

    def test_query_unauthenticated(self):
        resp = client.post("/api/patients/test/query", json={
            "message": "Hello", "session_id": "test",
        })
        assert resp.status_code == 401

    def test_query_out_of_scope(self, token):
        resp = client.post("/api/patients/test/query", json={
            "message": "Write a python program to sort a list",
            "session_id": "api-test-oos",
        }, headers=self._headers(token))
        assert resp.status_code == 200
        data = resp.json()
        assert data["intent"] == "out_of_scope"

    def test_root_unauthenticated(self):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_auth_register_unauthenticated(self):
        import uuid
        email = f"public-{uuid.uuid4().hex[:8]}@cliniq.com"
        resp = client.post("/api/auth/register", json={
            "email": email, "password": "test1234",
        })
        assert resp.status_code == 200

    def test_auth_login_unauthenticated(self):
        resp = client.post("/api/auth/login", json={
            "email": "public@cliniq.com", "password": "test1234",
        })
        assert resp.status_code in (200, 401)
