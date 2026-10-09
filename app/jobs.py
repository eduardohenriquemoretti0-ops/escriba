"""Modelo de job e store em memória com TTL/expurgo.

FIX M3: `warnings` (lista, não-fatal) separado de `error` (fatal) — cada campo
tem uma única semântica.
FIX M4/M5: store protegido por lock, com limpeza periódica (sem vazamento).
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    filename: str = ""
    transcript: str | None = None
    ata: str | None = None
    error: str | None = None                      # fatal (M3)
    warnings: list[str] = field(default_factory=list)  # não-fatal (M3)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()

    def public(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "filename": self.filename,
            "has_transcript": self.transcript is not None,
            "has_ata": self.ata is not None,
            "error": self.error,
            "warnings": self.warnings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobStore:
    def __init__(self, ttl_seconds: int = 7 * 86400):
        self.ttl = ttl_seconds
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> Job:
        job = Job(id=uuid.uuid4().hex, filename=filename)
        with self._lock:
            self._gc_locked()
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def _gc_locked(self) -> None:
        now = time.time()
        dead = [k for k, j in self._jobs.items() if (now - j.updated_at) > self.ttl]
        for k in dead:
            self._jobs.pop(k, None)
