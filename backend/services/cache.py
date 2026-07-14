import time
import asyncio
from collections import OrderedDict
from logger import get_logger

log = get_logger("cache")

MAX_SIZE = 200
TTL_SECONDS = 300

_cache: OrderedDict[str, tuple[float, object]] = OrderedDict()
_lock = asyncio.Lock()


async def get(key: str) -> object | None:
    async with _lock:
        if key not in _cache:
            return None
        ts, val = _cache[key]
        if time.monotonic() - ts > TTL_SECONDS:
            del _cache[key]
            return None
        _cache.move_to_end(key)
        return val


async def set(key: str, value: object) -> None:
    async with _lock:
        if key in _cache:
            _cache.move_to_end(key)
        else:
            if len(_cache) >= MAX_SIZE:
                _cache.popitem(last=False)
        _cache[key] = (time.monotonic(), value)


async def invalidate(prefix: str) -> None:
    async with _lock:
        keys = [k for k in _cache if k.startswith(prefix)]
        for k in keys:
            del _cache[k]
        if keys:
            log.debug(f"Invalidated {len(keys)} cache entries for prefix={prefix!r}")
