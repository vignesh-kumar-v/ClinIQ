from config import chroma_client
from logger import get_logger

log = get_logger("user_store")

_users_col = chroma_client.get_or_create_collection("users")


def create_user(email: str, password_hash: str, name: str = "") -> str:
    existing = _users_col.get(where={"email": {"$eq": email}})
    if existing["ids"]:
        raise ValueError("Email already registered")

    import uuid
    user_id = str(uuid.uuid4())
    _users_col.add(
        ids=[user_id],
        documents=[email],
        metadatas=[{
            "user_id": user_id,
            "email": email,
            "password_hash": password_hash,
            "name": name,
        }],
    )
    log.info(f"Created user {user_id} email={email}")
    return user_id


def get_user_by_email(email: str) -> dict | None:
    results = _users_col.get(
        where={"email": {"$eq": email}},
        include=["metadatas"],
    )
    if results["ids"]:
        return results["metadatas"][0]
    return None


def get_user_by_id(user_id: str) -> dict | None:
    results = _users_col.get(
        ids=[user_id],
        include=["metadatas"],
    )
    if results["ids"]:
        return results["metadatas"][0]
    return None
