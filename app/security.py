"""Segurança: comparação constante de token, validação de job_id, rate limit.

Implementa os fixes do escriba:
- M1: comparação de token constant-time (`hmac.compare_digest`).
- M2: validação de `job_id` por whitelist (anti path-traversal).
- C4 + M5: rate limiter por IP+endpoint, com TTL/limpeza (sem memory leak).
"""
from __future__ import annotations

import hmac
import re
import threading
import time
from collections import deque

_JOB_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def valid_job_id(job_id: str) -> bool:
    """M2: só aceita id em whitelist estrita (nunca usar input cru em path)."""
    return bool(_JOB_ID.match(job_id or ""))


def token_matches(provided: str, expected: str) -> bool:
    """M1: comparação constante no tempo; nunca `==` para segredos."""
    if not expected:
        return False
    return hmac.compare_digest((provided or "").encode(), expected.encode())


class RateLimiter:
    """C4/M5: janela deslizante por chave (IP+endpoint). Limpeza embutida."""

    def __init__(self, max_requests: int, window_sec: int):
        self.max = max_requests
        self.window = window_sec
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._last_gc = 0.0

    def _gc(self, now: float) -> None:
        # Remove chaves sem hits recentes (evita crescimento infinito do dict).
        if now - self._last_gc < self.window:
            return
        self._last_gc = now
        dead = [
            k for k, dq in self._hits.items()
            if not dq or (now - dq[-1]) > self.window
        ]
        for k in dead:
            self._hits.pop(k, None)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            self._gc(now)
            dq = self._hits.setdefault(key, deque())
            while dq and (now - dq[0]) > self.window:
                dq.popleft()
            if len(dq) >= self.max:
                return False
            dq.append(now)
            return True
