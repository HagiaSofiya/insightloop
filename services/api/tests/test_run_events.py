import asyncio
import json

import asyncpg
import pytest

from app import run_events
from app.db import connect
from app.run_events import BATCH, emit, events_after, format_sse, stream_events
from app.workflow import Activities, advance, cancel, pause, resume, retry, start
from dbtools import in_rolled_back_transaction
from test_workflow import Fake, new_run


def parse(messages):
    """SSE text -> list of (id, type, data)."""
    out = []
    for message in messages:
        if message.startswith(":"):
            continue
        fields = dict(line.split(": ", 1) for line in message.strip().split("\n"))
        out.append((int(fields["id"]), fields["event"], json.loads(fields["data"])))
    return out


async def collect(conn, run_id, last_id=0, limit=1000):
    """Read a stream to its end using the test's own connection."""

    async def same_conn():
        return NoClose(conn)

    return parse([m async for m in stream_events(same_conn, run_id, last_id, poll_seconds=0)][:limit])


class NoClose:
    """Lets a test hand its transaction's connection to the stream without the stream closing it."""

    def __init__(self, conn):
        self.conn = conn

    def __getattr__(self, name):
        return getattr(self.conn, name)

    async def close(self):
        pass


def test_format_sse_is_what_the_browser_expects():
    assert format_sse(7, "step.started", '{"step": "guide"}') == 'id: 7\nevent: step.started\ndata: {"step": "guide"}\n\n'


def test_a_full_run_leaves_an_ordered_log_the_stream_can_replay():
    async def body(conn):
        run_id, fake = await new_run(conn, count=3), Fake()
        await advance(conn, run_id, fake.activities, concurrency=1)
        events = await collect(conn, run_id)
        types = [t for _, t, _ in events]
        assert types[0] == "run.created" and types[-1] == "run.done"
        assert types.count("respondent.saved") == 3 and types.count("step.finished") == 6
        assert [d["step"] for _, t, d in events if t == "step.started"] == ["guide", "review", "respondents", "embed", "themes", "synthesize"]
        ids = [i for i, _, _ in events]
        assert ids == sorted(ids) and len(set(ids)) == len(ids)
        progress = [d["completed"] for _, t, d in events if t == "respondent.saved"]
        assert progress == [1, 2, 3]

    in_rolled_back_transaction(body)


def test_reconnecting_with_the_last_id_replays_exactly_what_was_missed():
    async def body(conn):
        run_id, fake = await new_run(conn, count=3), Fake()
        await advance(conn, run_id, fake.activities, concurrency=1)
        everything = await collect(conn, run_id)
        seen = everything[:6]  # the client saw six events, then the connection dropped
        replay = await collect(conn, run_id, last_id=seen[-1][0])
        assert seen + replay == everything  # no gap and no duplicate
        assert await collect(conn, run_id, last_id=everything[-1][0]) == []  # nothing left to replay

    in_rolled_back_transaction(body)


def test_control_calls_and_failures_appear_in_the_log():
    async def body(conn):
        run_id, fake = await new_run(conn), Fake()
        fake.embed_failures = 1
        with pytest.raises(RuntimeError):
            await advance(conn, run_id, fake.activities, concurrency=1)
        await pause(conn, run_id)
        await resume(conn, run_id)
        await cancel(conn, run_id)
        types = [t for _, t, _ in await collect(conn, run_id)]
        failed = [d for _, t, d in await collect(conn, run_id) if t == "step.failed"]
        assert "step.failed" in types and failed[0]["step"] == "embed" and "embeddings API down" in failed[0]["error"]
        assert types[-3:] == ["run.paused", "run.resumed", "run.cancelled"]

    in_rolled_back_transaction(body)


def test_the_stream_ends_when_the_run_is_finished_but_not_when_it_is_merely_paused():
    async def body(conn):
        run_id = await new_run(conn)
        await pause(conn, run_id)
        stream = stream_events(lambda: asyncio.sleep(0, NoClose(conn)), run_id, 0, poll_seconds=0)
        got = [await stream.__anext__(), await stream.__anext__()]  # created, paused
        assert [m.split("\n")[1] for m in got] == ["event: run.created", "event: run.paused"]
        waiting = asyncio.ensure_future(stream.__anext__())
        await asyncio.sleep(0.05)
        assert not waiting.done()  # paused is not the end: it waits for more
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await stream.aclose()
        await cancel(conn, run_id)
        assert [t for _, t, _ in await collect(conn, run_id)][-1] == "run.cancelled"  # cancelled is the end: the stream finishes

    in_rolled_back_transaction(body)


def test_a_backlog_longer_than_one_page_is_sent_in_full_and_in_order():
    async def body(conn):
        run_id = await new_run(conn)
        for n in range(BATCH + 30):
            await emit(conn, run_id, "tick", {"n": n})
        await cancel(conn, run_id)
        events = await collect(conn, run_id)
        ticks = [d["n"] for _, t, d in events if t == "tick"]
        assert ticks == list(range(BATCH + 30))

    in_rolled_back_transaction(body)


# ---- disconnects and cleanup -------------------------------------------------------------------------


class Tracked:
    """A stand-in connection factory that records whether the stream closed its connection."""

    def __init__(self, real):
        self.real, self.closed = real, False

    async def __call__(self):
        outer = self

        class Conn(NoClose):
            async def close(self):
                outer.closed = True

        return Conn(self.real)


def test_the_connection_is_closed_when_the_client_disconnects():
    async def body(conn):
        run_id = await new_run(conn)
        tracked, checks = Tracked(conn), {"n": 0}

        async def disconnected():
            checks["n"] += 1
            return checks["n"] > 2

        messages = [m async for m in stream_events(tracked, run_id, 0, disconnected, poll_seconds=0)]
        assert len(messages) == 1 and tracked.closed  # sent what existed, saw the disconnect, cleaned up

    in_rolled_back_transaction(body)


def test_the_connection_is_closed_when_the_server_cancels_the_stream_mid_wait():
    async def body(conn):
        run_id = await new_run(conn)
        tracked = Tracked(conn)
        stream = stream_events(tracked, run_id, 0, None, poll_seconds=10)
        await stream.__anext__()  # created
        waiting = asyncio.ensure_future(stream.__anext__())  # now sleeping for 10 seconds
        await asyncio.sleep(0.05)
        waiting.cancel()  # what the server does when the socket closes
        with pytest.raises(asyncio.CancelledError):
            await waiting
        await stream.aclose()
        assert tracked.closed

    in_rolled_back_transaction(body)


def test_idle_streams_send_a_keepalive_comment():
    async def body(conn):
        run_id = await new_run(conn)
        stream = stream_events(lambda: asyncio.sleep(0, NoClose(conn)), run_id, 0, poll_seconds=0.01, keepalive_seconds=0.02)
        first = await stream.__anext__()
        second = await asyncio.wait_for(stream.__anext__(), 1)
        await stream.aclose()
        assert first.startswith("id:") and second == ": keepalive\n\n"

    in_rolled_back_transaction(body)


# ---- ids follow commit order -------------------------------------------------------------------------------


def test_two_writers_on_the_same_run_cannot_commit_out_of_id_order():
    """The reason emit takes a lock: without it id 8 could commit before id 7 and a reader would skip 7 for good."""

    async def run():
        try:
            a, b = await connect(), await connect()
        except (OSError, asyncpg.PostgresError):
            pytest.skip("Postgres is not running (start it with make up)")
        run_id = await start(a, "lock test", 1, 0, "m")
        try:
            txn = a.transaction()
            await txn.start()
            first = await emit(a, run_id, "from.a")  # a holds the run's emit lock until it commits
            second = asyncio.ensure_future(emit(b, run_id, "from.b"))
            await asyncio.sleep(0.2)
            assert not second.done()  # b must wait, so it cannot take a later id and commit first
            await txn.commit()
            assert await asyncio.wait_for(second, 5) > first
        finally:
            await a.execute("DELETE FROM events WHERE type = 'workflow.advance' AND payload->>'run_id' = $1", str(run_id))
            await a.execute("DELETE FROM workflow_runs WHERE id = $1", run_id)
            await a.close()
            await b.close()

    asyncio.run(run())
