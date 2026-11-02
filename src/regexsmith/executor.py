"""Regex execution with a hard per-pattern timeout."""

from __future__ import annotations

import enum
import json
import math
import os
import select
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

MUST_ACCEPT, MUST_REJECT = 1, 2
BOTH = MUST_ACCEPT | MUST_REJECT
_WORKER = Path(__file__).with_name("_regex_worker.py")
_STARTUP_TIMEOUT = 30.0


class Verdict(enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    TIMEOUT = "timeout"
    ERROR = "error"


_CODES = {b"1": Verdict.PASS, b"0": Verdict.FAIL, b"e": Verdict.ERROR}


class BudgetExceeded(Exception):
    """The caller's overall deadline passed while a batch was running."""


class RegexRunner:
    """Checks batches of patterns against examples in a killable subprocess.

    A query that does not answer within ``timeout`` gets a TIMEOUT verdict, the worker is
    killed and replaced, and the rest of the batch carries on. Callers never see a hang.
    A batch of slow patterns can still add up, so ``check`` also takes an absolute deadline.
    """

    def __init__(self, timeout: float = 0.25) -> None:
        self.timeout = timeout
        self.queries_run = 0
        self.timeouts = 0
        self._process: subprocess.Popen | None = None

    def __enter__(self) -> RegexRunner:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._process is not None:
            self._process.kill()
            self._process.wait()
            self._process.stdin.close()
            self._process.stdout.close()
            self._process = None

    def _worker(self) -> subprocess.Popen:
        if self._process is None:
            self._process = subprocess.Popen(
                [sys.executable, "-I", "-S", str(_WORKER)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            ready, _, _ = select.select([self._process.stdout.fileno()], [], [], _STARTUP_TIMEOUT)
            if not ready or not os.read(self._process.stdout.fileno(), 64).startswith(b"ready"):
                self.close()
                raise RuntimeError("regex worker failed to start")
        return self._process

    def check(
        self,
        queries: Sequence[tuple[str, int]],
        positives: Sequence[str],
        negatives: Sequence[str],
        timeout: float | None = None,
        deadline: float = math.inf,
    ) -> list[Verdict]:
        """Run ``(pattern, mode)`` queries, where mode says which examples to enforce.

        ``timeout`` overrides the per-query limit for this call. Raises BudgetExceeded once
        ``deadline``, in ``time.perf_counter`` terms, has passed.
        """
        limit = self.timeout if timeout is None else timeout
        verdicts: list[Verdict] = []
        while len(verdicts) < len(queries):
            pending = queries[len(verdicts) :]
            answered, hung = self._run_until_hang(pending, positives, negatives, limit, deadline)
            verdicts.extend(answered)
            if hung:
                self.close()
                if time.perf_counter() >= deadline:
                    raise BudgetExceeded
                verdicts.append(Verdict.TIMEOUT)
                self.timeouts += 1
        self.queries_run += len(queries)
        return verdicts

    def _run_until_hang(
        self,
        queries: Sequence[tuple[str, int]],
        positives: Sequence[str],
        negatives: Sequence[str],
        limit: float,
        deadline: float,
    ) -> tuple[list[Verdict], bool]:
        process = self._worker()
        message = {"pos": list(positives), "neg": list(negatives), "queries": list(queries)}
        # The worker reads the whole line before it writes anything, so a large batch
        # cannot deadlock against a full output pipe.
        process.stdin.write(json.dumps(message).encode() + b"\n")
        process.stdin.flush()

        fd = process.stdout.fileno()
        verdicts: list[Verdict] = []
        buffer = b""
        query_deadline = time.perf_counter() + limit
        while len(verdicts) < len(queries):
            wait = min(query_deadline, deadline) - time.perf_counter()
            ready, _, _ = select.select([fd], [], [], max(wait, 0.0))
            if not ready:
                return verdicts, True
            chunk = os.read(fd, 65536)
            if not chunk:
                return verdicts, True
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            if lines:
                verdicts.extend(_CODES[line] for line in lines)
                query_deadline = time.perf_counter() + limit
        return verdicts, False
