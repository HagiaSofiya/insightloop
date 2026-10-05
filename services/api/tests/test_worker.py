import asyncio
from types import SimpleNamespace as NS

import pytest

from app.events import enqueue
from app.research_jobs import COMPLETED, REQUESTED, get, handle_requested, mark_failed, submit
from app.worker import Handler, work_once
from dbtools import in_rolled_back_transaction


class FakeResearch:
    def __init__(self, answer="the report", fail_times=0):
        self.calls, self.fail_times, self.answer = [], fail_times, answer

    async def __call__(self, question):
        self.calls.append(question)
        if len(self.calls) <= self.fail_times:
            raise RuntimeError("model unavailable")
        return NS(answer=self.answer, verification=NS(verified=True))


def handlers(fake):
    return {
        REQUESTED: Handler(lambda conn, payload: handle_requested(conn, payload, fake), mark_failed),
        COMPLETED: Handler(lambda conn, payload: asyncio.sleep(0)),
    }


async def events(conn, type=None):
    return await conn.fetch("SELECT * FROM events WHERE ($1::text IS NULL OR type = $1) ORDER BY id", type)


async def make_due(conn):
    await conn.execute("UPDATE events SET available_at = clock_timestamp() WHERE status = 'pending'")


async def work_all(conn, fake, **kwargs):
    while await work_once(conn, handlers(fake), **kwargs):
        pass


# ---- the outbox: request and event are written together --------------------------------------------


def test_submit_writes_the_request_and_its_event_together():
    async def body(conn):
        request_id = await submit(conn, "why did they stop?")
        request = await get(conn, request_id)
        assert (request["status"], request["question"]) == ("queued", "why did they stop?")
        [event] = [e for e in await events(conn, REQUESTED) if f'"request_id": {request_id}' in e["payload"]]
        assert event["status"] == "pending"

    in_rolled_back_transaction(body)


def test_if_the_surrounding_transaction_fails_neither_the_request_nor_the_event_exists():
    async def body(conn):
        before = (await conn.fetchval("SELECT count(*) FROM research_requests"), await conn.fetchval("SELECT count(*) FROM events"))
        with pytest.raises(RuntimeError):
            async with conn.transaction():
                await submit(conn, "q")
                raise RuntimeError("the API crashed after writing")
        after = (await conn.fetchval("SELECT count(*) FROM research_requests"), await conn.fetchval("SELECT count(*) FROM events"))
        assert after == before

    in_rolled_back_transaction(body)


# ---- the worker --------------------------------------------------------------------------------


def test_the_worker_runs_the_pipeline_saves_the_answer_and_records_a_completed_event():
    async def body(conn):
        request_id = await submit(conn, "q?")
        fake = FakeResearch()
        await work_all(conn, fake)
        request = await get(conn, request_id)
        assert (request["status"], request["answer"], request["verified"]) == ("done", "the report", True)
        assert fake.calls == ["q?"]
        completed = [e for e in await events(conn, COMPLETED) if f'"request_id": {request_id}' in e["payload"]]
        assert len(completed) == 1 and completed[0]["status"] == "done"  # the fact was recorded, then logged as handled
        assert {e["status"] for e in await events(conn, REQUESTED) if f'"request_id": {request_id}' in e["payload"]} == {"done"}

    in_rolled_back_transaction(body)


def test_delivering_the_same_event_twice_does_the_work_once():
    async def body(conn):
        request_id = await submit(conn, "q?")
        await enqueue(conn, REQUESTED, {"request_id": request_id})  # a duplicate delivery of the same request
        fake = FakeResearch()
        await work_all(conn, fake)
        assert fake.calls == ["q?"]  # the second delivery saw 'done' and skipped
        completed = [e for e in await events(conn, COMPLETED) if f'"request_id": {request_id}' in e["payload"]]
        assert len(completed) == 1
        assert (await get(conn, request_id))["answer"] == "the report"

    in_rolled_back_transaction(body)


def test_a_finished_request_is_never_overwritten_by_a_late_delivery():
    async def body(conn):
        request_id = await submit(conn, "q?")
        await work_all(conn, FakeResearch(answer="first"))
        await enqueue(conn, REQUESTED, {"request_id": request_id})
        later = FakeResearch(answer="second")
        await work_all(conn, later)
        assert later.calls == [] and (await get(conn, request_id))["answer"] == "first"

    in_rolled_back_transaction(body)


def test_a_failure_is_retried_and_the_second_delivery_succeeds():
    async def body(conn):
        request_id = await submit(conn, "q?")
        fake = FakeResearch(fail_times=1)
        await work_all(conn, fake)  # first delivery fails and goes back to pending, waiting out its backoff
        assert (await get(conn, request_id))["status"] == "running"
        await make_due(conn)
        await work_all(conn, fake)
        assert (await get(conn, request_id))["status"] == "done" and len(fake.calls) == 2

    in_rolled_back_transaction(body)


def test_an_event_that_keeps_failing_is_dead_lettered_and_the_request_marked_failed():
    async def body(conn):
        request_id = await submit(conn, "q?")
        fake = FakeResearch(fail_times=99)
        for _ in range(2):
            await work_all(conn, fake, max_attempts=2)
            await make_due(conn)
        request = await get(conn, request_id)
        assert request["status"] == "failed" and "model unavailable" in request["error"]
        [event] = [e for e in await events(conn, REQUESTED) if f'"request_id": {request_id}' in e["payload"]]
        assert (event["status"], event["attempts"]) == ("dead", 2)
        assert len(fake.calls) == 2  # not tried again once dead

    in_rolled_back_transaction(body)


def test_an_event_whose_worker_died_is_picked_up_again_after_its_lease_expires():
    async def body(conn):
        from app.events import claim

        request_id = await submit(conn, "q?")
        crashed = await claim(conn, [REQUESTED], lease_seconds=60)  # a worker claims it, then is killed
        assert crashed is not None
        fake = FakeResearch()
        await work_all(conn, fake)
        assert fake.calls == [] and (await get(conn, request_id))["status"] == "queued"  # still leased: nobody touches it
        await conn.execute("UPDATE events SET locked_until = clock_timestamp() - interval '1 second' WHERE id = $1", crashed.id)
        await work_all(conn, fake)
        assert (await get(conn, request_id))["status"] == "done" and fake.calls == ["q?"]

    in_rolled_back_transaction(body)


def test_an_event_that_killed_its_worker_every_time_is_given_up_on_without_running_it_again():
    async def body(conn):
        request_id = await submit(conn, "q?")
        event_id = await conn.fetchval("SELECT id FROM events WHERE type = $1 ORDER BY id DESC LIMIT 1", REQUESTED)
        await conn.execute(
            "UPDATE events SET status = 'processing', attempts = 3, locked_until = clock_timestamp() - interval '1 second' WHERE id = $1",
            event_id,
        )
        fake = FakeResearch()
        await work_all(conn, fake, max_attempts=3)
        assert fake.calls == []
        assert (await get(conn, request_id))["status"] == "failed"
        assert (await conn.fetchval("SELECT status FROM events WHERE id = $1", event_id)) == "dead"

    in_rolled_back_transaction(body)


def test_an_unknown_event_type_is_dead_lettered_not_retried_forever():
    async def body(conn):
        event_id = await enqueue(conn, "something.else", {})
        await work_all(conn, FakeResearch())
        assert (await conn.fetchval("SELECT status FROM events WHERE id = $1", event_id)) == "dead"

    in_rolled_back_transaction(body)
