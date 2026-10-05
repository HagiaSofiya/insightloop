"""Research as a background job: the API records a request and an event together, the worker runs the pipeline.

The handler is idempotent: delivering the same event twice (or again after a crash) finishes the request once
and never overwrites a finished result.
"""

import logging
from collections.abc import Awaitable, Callable

import asyncpg

from app.events import enqueue
from app.multi_agent import Research, research_question

logger = logging.getLogger(__name__)

REQUESTED = "research.requested"  # a command-like event: "please run this"
COMPLETED = "research.completed"  # a fact: "this finished". Nothing consumes it yet; M13 streams it.

ResearchFn = Callable[[str], Awaitable[Research]]


async def submit(conn: asyncpg.Connection, question: str) -> int:
    """Create the request and its event in ONE transaction, so neither can exist without the other."""
    async with conn.transaction():
        request_id = await conn.fetchval("INSERT INTO research_requests (question) VALUES ($1) RETURNING id", question)
        await enqueue(conn, REQUESTED, {"request_id": request_id})
    return request_id


async def get(conn: asyncpg.Connection, request_id: int) -> asyncpg.Record | None:
    return await conn.fetchrow(
        "SELECT id, question, status, answer, verified, error, created_at, updated_at FROM research_requests WHERE id = $1", request_id
    )


async def handle_requested(conn: asyncpg.Connection, payload: dict, research_fn: ResearchFn = research_question) -> None:
    request_id = payload["request_id"]
    # Only a queued or already-running request may start (running means an earlier delivery died part-way).
    row = await conn.fetchrow(
        "UPDATE research_requests SET status = 'running', updated_at = now() "
        "WHERE id = $1 AND status IN ('queued', 'running') RETURNING question",
        request_id,
    )
    if row is None:
        logger.info("research request %s is already finished or gone, skipping this delivery", request_id)
        return
    result = await research_fn(row["question"])
    async with conn.transaction():
        # The same guard on the way out: the first delivery to finish wins, a later one changes nothing.
        saved = await conn.fetchval(
            "UPDATE research_requests SET status = 'done', answer = $2, verified = $3, error = NULL, updated_at = now() "
            "WHERE id = $1 AND status = 'running' RETURNING id",
            request_id, result.answer, result.verification.verified,
        )
        if saved is not None:
            await enqueue(conn, COMPLETED, {"request_id": request_id, "verified": result.verification.verified})


async def mark_failed(conn: asyncpg.Connection, payload: dict, error: str) -> None:
    """Called when the event is dead-lettered, so the person who asked sees a failure instead of 'running' forever."""
    await conn.execute(
        "UPDATE research_requests SET status = 'failed', error = $2, updated_at = now() WHERE id = $1 AND status <> 'done'",
        payload["request_id"], error[:2000],
    )
