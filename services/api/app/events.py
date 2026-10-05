"""The event log: write an event, claim it, finish it, or fail it with a retry.

Delivery is at-least-once: an event can be handed out more than once (a worker may die after doing the work
but before marking it done), so handlers must be idempotent. Every function takes the caller's connection so
the caller decides the transaction: the API calls `enqueue` inside the transaction that changes its own state.
"""

import json
import random
from dataclasses import dataclass

import asyncpg

MAX_ATTEMPTS = 5  # after this many deliveries an event is dead-lettered
LEASE_SECONDS = 120  # how long a claim lasts before another worker may take the event over
BACKOFF_BASE = 2.0  # seconds; doubles each attempt, with full jitter
BACKOFF_CAP = 60.0


@dataclass
class Event:
    id: int
    type: str
    payload: dict
    attempts: int  # including the delivery that just claimed it


# One statement, so finding the row and locking it are atomic. SKIP LOCKED means a second worker running this
# at the same moment skips rows the first has locked and takes the next one, instead of waiting or taking the same row.
# An expired lease counts as claimable: that is how an event survives its worker being killed.
CLAIM_SQL = """
WITH next AS (
    SELECT id FROM events
    WHERE ((status = 'pending' AND available_at <= clock_timestamp())
           OR (status = 'processing' AND locked_until < clock_timestamp()))
      AND ($2::text[] IS NULL OR type = ANY($2))
    ORDER BY id
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
UPDATE events e
SET status = 'processing', attempts = e.attempts + 1,
    locked_until = clock_timestamp() + $1 * interval '1 second'
FROM next WHERE e.id = next.id
RETURNING e.id, e.type, e.payload, e.attempts
"""


async def enqueue(conn: asyncpg.Connection, type: str, payload: dict) -> int:
    """Write an event. Call this inside the transaction that changes the state the event is about."""
    return await conn.fetchval("INSERT INTO events (type, payload) VALUES ($1, $2::jsonb) RETURNING id", type, json.dumps(payload))


async def claim(conn: asyncpg.Connection, types: list[str] | None = None, lease_seconds: float = LEASE_SECONDS) -> Event | None:
    row = await conn.fetchrow(CLAIM_SQL, lease_seconds, types)
    return Event(row["id"], row["type"], json.loads(row["payload"]), row["attempts"]) if row else None


async def complete(conn: asyncpg.Connection, event_id: int) -> None:
    await conn.execute(
        "UPDATE events SET status = 'done', processed_at = clock_timestamp(), locked_until = NULL, last_error = NULL "
        "WHERE id = $1 AND status = 'processing'",
        event_id,
    )


def backoff_seconds(attempts: int) -> float:
    """How long to wait before delivery number attempts + 1: exponential, capped, with full jitter."""
    return random.uniform(0, min(BACKOFF_CAP, BACKOFF_BASE * 2 ** (attempts - 1)))


async def fail(conn: asyncpg.Connection, event: Event, error: str, max_attempts: int = MAX_ATTEMPTS) -> bool:
    """Record a failed delivery. Returns True if the event is now dead (no more retries)."""
    dead = event.attempts >= max_attempts
    await conn.execute(
        """
        UPDATE events SET status = $2, last_error = $3, locked_until = NULL,
               available_at = clock_timestamp() + $4 * interval '1 second'
        WHERE id = $1 AND status = 'processing'
        """,
        event.id, "dead" if dead else "pending", error[:2000], 0.0 if dead else backoff_seconds(event.attempts),
    )
    return dead
