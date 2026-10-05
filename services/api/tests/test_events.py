import asyncio
import json

import asyncpg
import pytest

from app import events
from app.db import connect
from app.events import CLAIM_SQL, backoff_seconds, claim, complete, enqueue, fail
from dbtools import in_rolled_back_transaction


def test_enqueue_stores_a_pending_event():
    async def body(conn):
        event_id = await enqueue(conn, "t.created", {"x": 1})
        row = await conn.fetchrow("SELECT * FROM events WHERE id = $1", event_id)
        assert (row["type"], json.loads(row["payload"]), row["status"], row["attempts"]) == ("t.created", {"x": 1}, "pending", 0)

    in_rolled_back_transaction(body)


def test_claim_hands_out_events_in_order_and_counts_the_attempt():
    async def body(conn):
        first = await enqueue(conn, "t", {"n": 1})
        second = await enqueue(conn, "t", {"n": 2})
        a = await claim(conn, ["t"])
        b = await claim(conn, ["t"])
        assert (a.id, a.attempts, a.payload) == (first, 1, {"n": 1})
        assert b.id == second
        assert await claim(conn, ["t"]) is None  # both are leased now

    in_rolled_back_transaction(body)


def test_a_claimed_event_is_not_handed_out_again_until_its_lease_expires():
    async def body(conn):
        event_id = await enqueue(conn, "t", {})
        assert (await claim(conn, ["t"], lease_seconds=60)).id == event_id
        assert await claim(conn, ["t"]) is None
        await conn.execute("UPDATE events SET locked_until = clock_timestamp() - interval '1 second' WHERE id = $1", event_id)  # the worker died
        again = await claim(conn, ["t"])
        assert (again.id, again.attempts) == (event_id, 2)

    in_rolled_back_transaction(body)


def test_complete_marks_it_done_and_it_is_never_claimed_again():
    async def body(conn):
        event_id = await enqueue(conn, "t", {})
        await claim(conn, ["t"])
        await complete(conn, event_id)
        row = await conn.fetchrow("SELECT status, processed_at FROM events WHERE id = $1", event_id)
        assert row["status"] == "done" and row["processed_at"] is not None
        assert await claim(conn, ["t"]) is None

    in_rolled_back_transaction(body)


def test_a_failed_event_waits_for_its_backoff_then_is_retried():
    async def body(conn):
        event_id = await enqueue(conn, "t", {})
        event = await claim(conn, ["t"])
        assert await fail(conn, event, "boom") is False
        row = await conn.fetchrow("SELECT status, last_error FROM events WHERE id = $1", event_id)
        assert (row["status"], row["last_error"]) == ("pending", "boom")
        await conn.execute("UPDATE events SET available_at = clock_timestamp() + interval '1 hour' WHERE id = $1", event_id)
        assert await claim(conn, ["t"]) is None  # still backing off
        await conn.execute("UPDATE events SET available_at = clock_timestamp() WHERE id = $1", event_id)
        assert (await claim(conn, ["t"])).attempts == 2

    in_rolled_back_transaction(body)


def test_after_max_attempts_the_event_is_dead_and_not_retried():
    async def body(conn):
        event_id = await enqueue(conn, "t", {})
        for expected in (1, 2, 3):
            event = await claim(conn, ["t"])
            assert event.attempts == expected
            dead = await fail(conn, event, f"error {expected}", max_attempts=3)
            await conn.execute("UPDATE events SET available_at = clock_timestamp() WHERE id = $1", event_id)
        assert dead is True
        row = await conn.fetchrow("SELECT status, last_error FROM events WHERE id = $1", event_id)
        assert (row["status"], row["last_error"]) == ("dead", "error 3")
        assert await claim(conn, ["t"]) is None

    in_rolled_back_transaction(body)


def test_claim_only_takes_the_types_it_is_asked_for():
    async def body(conn):
        await enqueue(conn, "other", {})
        assert await claim(conn, ["wanted"]) is None
        assert (await claim(conn, None)).type == "other"  # no filter: anything

    in_rolled_back_transaction(body)


def test_backoff_grows_and_is_capped():
    assert all(0 <= backoff_seconds(1) <= 2 for _ in range(50))
    assert all(0 <= backoff_seconds(3) <= 8 for _ in range(50))
    assert all(0 <= backoff_seconds(30) <= events.BACKOFF_CAP for _ in range(50))


def test_two_workers_claiming_at_once_get_different_events():
    """The real SKIP LOCKED check: two connections, one holding a row lock inside an open transaction."""

    async def run():
        try:
            a, b = await connect(), await connect()
        except (OSError, asyncpg.PostgresError):
            pytest.skip("Postgres is not running (start it with make up)")
        kind = "test.skip_locked"
        try:
            ids = [await enqueue(a, kind, {"n": n}) for n in (1, 2)]
            txn = a.transaction()
            await txn.start()
            held = await a.fetchrow(CLAIM_SQL, 60.0, [kind])  # worker A has claimed the first and has not committed
            taken = await claim(b, [kind])  # worker B must skip A's locked row instead of waiting or taking it
            await txn.rollback()
            assert held["id"] == ids[0] and taken.id == ids[1]
        finally:
            await b.execute("DELETE FROM events WHERE type = 'test.skip_locked'")
            await a.close()
            await b.close()

    asyncio.run(run())
