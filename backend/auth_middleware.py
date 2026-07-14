from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from auth import decode_access_token
from logger import get_logger

log = get_logger("auth_middleware")

PUBLIC_PATHS = {
    "/",
    "/auth",
    "/api/auth/register",
    "/api/auth/login",
    "/docs",
    "/openapi.json",
    "/redoc",
}


def _is_public(path: str) -> bool:
    if path in PUBLIC_PATHS:
        return True
    if path.startswith("/assets/") or path.startswith("/static/"):
        return True
    if path in ("/favicon.svg", "/icons.svg"):
        return True
    return False


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if _is_public(request.url.path):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

        token = auth_header[7:]
        payload = decode_access_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        request.state.user_id = payload["sub"]
        request.state.user_email = payload.get("email", "")
        return await call_next(request)
