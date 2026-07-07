from fastapi import APIRouter, HTTPException
from models import RegisterRequest, LoginRequest, TokenResponse
from auth import hash_password, verify_password, create_access_token
from user_store import create_user, get_user_by_email
from logger import get_logger

log = get_logger("router.auth")
router = APIRouter()


@router.post("/auth/register", response_model=TokenResponse)
async def register(body: RegisterRequest):
    try:
        user_id = create_user(body.email, hash_password(body.password), body.name)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    token = create_access_token(user_id, body.email)
    log.info(f"User registered: {body.email}")
    return TokenResponse(
        access_token=token,
        user_id=user_id,
        email=body.email,
        name=body.name,
    )


@router.post("/auth/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    user = get_user_by_email(body.email)
    if not user or not verify_password(body.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user_id = user["user_id"]
    token = create_access_token(user_id, body.email)
    log.info(f"User logged in: {body.email}")
    return TokenResponse(
        access_token=token,
        user_id=user_id,
        email=body.email,
        name=user.get("name", ""),
    )
