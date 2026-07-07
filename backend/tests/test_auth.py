import pytest
from auth import hash_password, verify_password, create_access_token, decode_access_token


class TestPasswordHashing:
    def test_hash_and_verify(self):
        hashed = hash_password("mypassword")
        assert hashed != "mypassword"
        assert verify_password("mypassword", hashed)

    def test_wrong_password(self):
        hashed = hash_password("correct")
        assert not verify_password("wrong", hashed)

    def test_different_hashes(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2
        assert verify_password("same", h1)
        assert verify_password("same", h2)


class TestJWT:
    def test_create_and_decode(self):
        token = create_access_token("user-123", "test@test.com")
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "user-123"
        assert payload["email"] == "test@test.com"

    def test_invalid_token(self):
        assert decode_access_token("not.a.token") is None

    def test_empty_token(self):
        assert decode_access_token("") is None

    def test_tampered_token(self):
        token = create_access_token("user-123", "test@test.com")
        tampered = token[:-5] + "xxxxx"
        assert decode_access_token(tampered) is None
