import time
import asyncio
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from logger import get_logger

log = get_logger("ratelimit")

_WINDOW_SECONDS = 60
_MAX_REQUESTS = 30
_LLM_MAX_REQUESTS = 10

_buckets: dict[str, list[float]] = defaultdict(list)
_llm_buckets: dict[str, list[float]] = defaultdict(list)
_lock = asyncio.Lock()

_LLM_PATH_SUFFIXES = {"/query", "/query/stream"}


def _clean_bucket(bucket: list[float], window: float) -> list[float]:
    now = time.monotonic()
    return [t for t in bucket if now - t < window]


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        is_llm = any(path.endswith(s) for s in _LLM_PATH_SUFFIXES)
        max_req = _LLM_MAX_REQUESTS if is_llm else _MAX_REQUESTS

        async with _lock:
            bucket = _llm_buckets if is_llm else _buckets
            bucket[client_ip] = _clean_bucket(bucket[client_ip], _WINDOW_SECONDS)
            if len(bucket[client_ip]) >= max_req:
                log.warning(f"Rate limit hit: ip={client_ip} path={path} count={len(bucket[client_ip])}")
                raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
            bucket[client_ip].append(time.monotonic())

        return await call_next(request)
