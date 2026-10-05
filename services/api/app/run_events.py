"""The log of what happened in a workflow run, and the Server-Sent Events stream that serves it.

Replay works because every event has an increasing id and a client reconnects with the last id it saw
(the Last-Event-ID header): the server sends everything after it. That is only correct if ids appear in the
order they commit. A plain bigserial does not promise that (id 7 can commit after id 8), so `emit` first takes
a lock for the run that lasts until the transaction ends, which makes ids follow commit order within a run.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

import asyncpg

logger = logging.getLogger(__name__)

EMIT_LOCK_NAMESPACE = 7013  # not the same key as the executor's lock, so the two never wait on each other
TERMINAL = ("done", "failed", "cancelled")
BATCH = 100  # events fetched per query, so a long backlog is sent in pieces and a slow client costs little memory
POLL_SECONDS = 0.25
KEEPALIVE_SECONDS = 15.0


async def emit(conn: asyncpg.Connection, run_id: int, type: str, data: dict | None = None) -> int:
    """Append an event. Call it last in the transaction that changes the state it describes."""
    async with conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock($1, $2)", EMIT_LOCK_NAMESPACE, run_id)
        return await conn.fetchval(
            "INSERT INTO run_events (run_id, type, data) VALUES ($1, $2, $3::jsonb) RETURNING id", run_id, type, json.dumps(data or {})
        )


async def events_after(conn: asyncpg.Connection, run_id: int, last_id: int, limit: int = BATCH) -> list[asyncpg.Record]:
    return await conn.fetch(
        "SELECT id, type, data, created_at FROM run_events WHERE run_id = $1 AND id > $2 ORDER BY id LIMIT $3", run_id, last_id, limit
    )


def format_sse(event_id: int, type: str, data: str) -> str:
    """One SSE message. The blank line ends it; `id` is what the browser sends back as Last-Event-ID."""
    return f"id: {event_id}\nevent: {type}\ndata: {data}\n\n"


async def stream_events(
    connect: Callable[[], Awaitable[asyncpg.Connection]],
    run_id: int,
    last_id: int = 0,
    disconnected: Callable[[], Awaitable[bool]] | None = None,
    poll_seconds: float = POLL_SECONDS,
    keepalive_seconds: float = KEEPALIVE_SECONDS,
) -> AsyncIterator[str]:
    """Yield SSE messages for a run: everything after last_id, then new events as they appear.

    Ends when the run is finished and everything is sent, or when the client goes away. The generator is only
    resumed when the server can write more, so a slow client slows this loop and never builds up a queue in memory:
    the backlog stays in the database, and the client's last id says where to pick up.
    """
    conn = await connect()  # this stream's own connection: it must be closed on every way out
    idle = 0.0
    try:
        while True:
            if disconnected is not None and await disconnected():
                return
            status = await conn.fetchval("SELECT status FROM workflow_runs WHERE id = $1", run_id)  # read BEFORE the events,
            events = await events_after(conn, run_id, last_id)  # so a finished run's last events are never missed
            for event in events:
                last_id = event["id"]
                idle = 0.0
                yield format_sse(event["id"], event["type"], event["data"])
            if len(events) == BATCH:
                continue  # more may be waiting: fetch the next page before sleeping
            if status is None or status in TERMINAL:
                return
            await asyncio.sleep(poll_seconds)
            idle += poll_seconds
            if idle >= keepalive_seconds:
                idle = 0.0
                yield ": keepalive\n\n"  # a comment line: keeps proxies from closing an idle connection
    finally:
        await conn.close()
