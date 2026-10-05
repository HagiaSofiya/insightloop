"""The worker: claim an event, run its handler outside any transaction, then mark it done or failed.

Killing the worker at any point loses nothing: an unfinished event's lease runs out and the next claim takes it.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import asyncpg

from app.db import connect
from app.events import LEASE_SECONDS, MAX_ATTEMPTS, claim, complete, fail
from app import workflow
from app.research_jobs import COMPLETED, REQUESTED, handle_requested, mark_failed

logger = logging.getLogger(__name__)

POLL_SECONDS = 1.0


@dataclass
class Handler:
    run: Callable[[asyncpg.Connection, dict], Awaitable[None]]
    on_dead: Callable[[asyncpg.Connection, dict, str], Awaitable[None]] | None = None


HANDLERS: dict[str, Handler] = {
    REQUESTED: Handler(handle_requested, mark_failed),
    workflow.ADVANCE: Handler(workflow.handle_advance, workflow.mark_failed),
    COMPLETED: Handler(lambda conn, payload: asyncio.sleep(0)),  # a log-only fact: nothing to do yet
}


async def work_once(
    conn: asyncpg.Connection, handlers: dict[str, Handler] = HANDLERS, lease_seconds: float = LEASE_SECONDS,
    max_attempts: int = MAX_ATTEMPTS,
) -> bool:
    """Handle at most one event. Returns False when there was nothing to claim."""
    event = await claim(conn, None, lease_seconds)
    if event is None:
        return False
    handler = handlers.get(event.type)
    if handler is None:
        logger.warning("event %s has unknown type %r: dead-lettering it", event.id, event.type)
        await fail(conn, event, f"no handler for {event.type}", max_attempts=1)
        return True
    error = None
    if event.attempts > max_attempts:  # every earlier delivery ran out its lease: the worker kept dying on it
        error = f"gave up after {event.attempts - 1} deliveries that never finished"
    else:
        try:
            await handler.run(conn, event.payload)
        except Exception as exc:  # any handler failure is retried; a bug should not kill the worker
            logger.exception("event %s (%s) failed on attempt %d", event.id, event.type, event.attempts)
            error = f"{type(exc).__name__}: {exc}"
    if error is None:
        await complete(conn, event.id)
        logger.info("event %s (%s) done", event.id, event.type)
        return True
    if await fail(conn, event, error, max_attempts) and handler.on_dead:
        logger.error("event %s (%s) is dead: %s", event.id, event.type, error)
        await handler.on_dead(conn, event.payload, error)
    return True


async def run_worker(
    lease_seconds: float = LEASE_SECONDS, poll_seconds: float = POLL_SECONDS, stop: asyncio.Event | None = None
) -> None:
    stop = stop or asyncio.Event()
    conn = await connect()
    try:
        while not stop.is_set():
            if not await work_once(conn, lease_seconds=lease_seconds):
                try:
                    await asyncio.wait_for(stop.wait(), poll_seconds)
                except asyncio.TimeoutError:
                    pass
    finally:
        await conn.close()
