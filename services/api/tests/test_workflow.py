import asyncio
import json

import asyncpg
import pytest

from app import run_events, workflow
from app.db import connect
from app.events import claim, enqueue
from app.worker import Handler, work_once
from app.workflow import (
    ADVANCE, STEPS, Activities, WorkflowError, advance, cancel, describe, handle_advance, mark_failed, pause, resume, retry, start,
)
from dbtools import in_rolled_back_transaction

QUESTIONS = ["Why try it?", "Why stop?"]


class Fake:
    """Fake activities that count their calls, and can fail or pause the run at chosen moments."""

    def __init__(self, conn=None):
        self.conn, self.calls = conn, {name: 0 for name in STEPS}
        self.fail_slots, self.embed_failures, self.on_respondent, self.on_guide, self.run_id = set(), 0, None, None, None
        self.respondent_calls = []

    @property
    def activities(self):
        return Activities(self.guide, self.respondent, self.embed, self.themes, self.synthesize)

    async def guide(self, brief, model):
        self.calls["guide"] += 1
        if self.on_guide:
            await self.on_guide()
        return QUESTIONS

    async def respondent(self, brief, profile, questions, model):
        self.calls["respondents"] += 1
        slot = profile["age"], profile["city"]
        self.respondent_calls.append(slot)
        if self.on_respondent:
            await self.on_respondent(len(self.respondent_calls))
        if len(self.respondent_calls) in self.fail_slots:
            raise RuntimeError("persona generation failed")
        return {"name": f"R{len(self.respondent_calls)}", "age": profile["age"], "gender": profile["gender"], "city": profile["city"],
                "income": profile["income"], "household": profile["household"], "occupation": "x", "backstory": "b", "speaking_style": "s"}, ["a1", "a2"]

    async def embed(self, conn, study_id):
        self.calls["embed"] += 1
        if self.embed_failures > 0:
            self.embed_failures -= 1
            raise RuntimeError("embeddings API down")

    async def themes(self, conn, study_id, questions):
        self.calls["themes"] += 1

    async def synthesize(self, conn, study_id, on_event):
        self.calls["synthesize"] += 1
        for piece in ("the ", "report"):
            await on_event("delta", piece)
        return "the report", {"plan": ["p"], "analysts": [], "critiques": [], "verification": {"verified": True, "problems": []}}


async def new_run(conn, count=4):
    return await start(conn, "a brief", count, 1, "m")


async def steps(conn, run_id):
    return {r["name"]: r["status"] for r in await conn.fetch("SELECT name, status FROM workflow_steps WHERE run_id = $1", run_id)}


async def run_status(conn, run_id):
    return await conn.fetchval("SELECT status FROM workflow_runs WHERE id = $1", run_id)


async def respondent_count(conn, run_id):
    return await conn.fetchval(
        "SELECT count(*) FROM respondents r JOIN workflow_runs w ON w.study_id = r.study_id WHERE w.id = $1", run_id)


# ---- the happy path and checkpoints ------------------------------------------------------------------


def test_a_run_goes_through_every_step_and_saves_each_ones_output():
    async def body(conn):
        run_id, fake = await new_run(conn), Fake()
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "done"
        assert set((await steps(conn, run_id)).values()) == {"done"}
        info = await describe(conn, run_id)
        assert info["report"] == "the report" and info["status"] == "done"
        by_name = {step["name"]: step for step in info["steps"]}
        assert by_name["guide"]["output"]["questions"] == QUESTIONS
        assert by_name["respondents"]["output"]["completed"] == 4
        assert by_name["synthesize"]["output"]["verification"]["verified"] is True  # the reasoning panel's data is stored with the step
        assert await respondent_count(conn, run_id) == 4
        assert await conn.fetchval("SELECT count(*) FROM responses s JOIN respondents r ON r.id = s.respondent_id WHERE r.study_id = $1", info["study_id"]) == 8

    in_rolled_back_transaction(body)


def test_start_writes_the_run_its_steps_and_the_event_together():
    async def body(conn):
        run_id = await new_run(conn)
        assert list((await steps(conn, run_id)).values()) == ["pending"] * 6
        event = await claim(conn, [ADVANCE])
        assert event.payload == {"run_id": run_id}

    in_rolled_back_transaction(body)


def test_a_crashed_run_resumes_at_the_failed_step_without_redoing_earlier_ones():
    async def body(conn):
        run_id, fake = await new_run(conn), Fake()
        fake.embed_failures = 1
        with pytest.raises(RuntimeError):  # the executor surfaces the failure so the event queue retries it
            await advance(conn, run_id, fake.activities, concurrency=1)
        assert await steps(conn, run_id) == {"guide": "done", "review": "done", "respondents": "done", "embed": "failed", "themes": "pending", "synthesize": "pending"}
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "done"
        assert fake.calls == {"guide": 1, "review": 0, "respondents": 4, "embed": 2, "themes": 1, "synthesize": 1}  # only embed ran twice

    in_rolled_back_transaction(body)


# ---- fan-out and partial failure ---------------------------------------------------------------------


def test_a_few_failed_respondents_are_recorded_and_the_step_still_passes():
    async def body(conn):
        run_id, fake = await new_run(conn, count=10), Fake()
        fake.fail_slots = {3, 7}  # 2 of 10 fail: 80% completed is enough
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "done"
        output = next(step for step in (await describe(conn, run_id))["steps"] if step["name"] == "respondents")["output"]
        assert output["completed"] == 8 and len(output["failed_slots"]) == 2
        assert "persona generation failed" in output["errors"][0]

    in_rolled_back_transaction(body)


def test_too_many_failures_fail_the_step_and_a_retry_redoes_only_the_missing_respondents():
    async def body(conn):
        run_id, fake = await new_run(conn, count=10), Fake()
        fake.fail_slots = set(range(1, 6))  # 5 of 10 fail
        with pytest.raises(workflow.StepFailed, match="only 5 of 10"):
            await advance(conn, run_id, fake.activities, concurrency=1)
        assert (await steps(conn, run_id))["respondents"] == "failed" and await respondent_count(conn, run_id) == 5
        fake.fail_slots, before = set(), len(fake.respondent_calls)
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "done"
        assert len(fake.respondent_calls) - before == 5  # the five that were missing, not all ten
        assert await respondent_count(conn, run_id) == 10

    in_rolled_back_transaction(body)


# ---- pause, resume, cancel ----------------------------------------------------------------------------


def test_pause_between_steps_stops_there_and_resume_continues():
    async def body(conn):
        run_id, fake = await new_run(conn), Fake()
        fake.on_guide = lambda: pause(conn, run_id)  # paused while the first step is working
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "paused"
        assert (await steps(conn, run_id))["guide"] == "done" and (await steps(conn, run_id))["respondents"] == "pending"
        assert fake.calls["respondents"] == 0
        await resume(conn, run_id)
        fake.on_guide = None
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "done" and fake.calls["guide"] == 1

    in_rolled_back_transaction(body)


def test_pause_in_the_middle_of_the_fan_out_keeps_finished_respondents_and_resume_does_only_the_rest():
    async def body(conn):
        run_id, fake = await new_run(conn, count=6), Fake()

        async def pause_after_two(n):
            if n == 2:
                await pause(conn, run_id)

        fake.on_respondent = pause_after_two
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "paused"
        assert await respondent_count(conn, run_id) == 2 and (await steps(conn, run_id))["respondents"] == "pending"
        await resume(conn, run_id)
        fake.on_respondent = None
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "done"
        assert fake.calls["respondents"] == 6 and await respondent_count(conn, run_id) == 6

    in_rolled_back_transaction(body)


def test_a_cancelled_run_stops_and_is_never_advanced_again():
    async def body(conn):
        run_id, fake = await new_run(conn), Fake()
        fake.on_guide = lambda: cancel(conn, run_id)
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "cancelled"
        calls = dict(fake.calls)
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "cancelled" and fake.calls == calls
        with pytest.raises(WorkflowError):
            await resume(conn, run_id)

    in_rolled_back_transaction(body)


def test_control_calls_reject_runs_in_the_wrong_state():
    async def body(conn):
        run_id, fake = await new_run(conn), Fake()
        await advance(conn, run_id, fake.activities, concurrency=1)  # now done
        for call in (pause(conn, run_id), cancel(conn, run_id), resume(conn, run_id)):
            with pytest.raises(WorkflowError):
                await call
        with pytest.raises(WorkflowError, match="No workflow run"):
            await pause(conn, 123456789)
        with pytest.raises(WorkflowError, match="Unknown step"):
            await retry(conn, run_id, "nope")

    in_rolled_back_transaction(body)


# ---- retry from a step (with compensation) ---------------------------------------------------------------


def test_retry_from_embed_undoes_embed_and_later_steps_only():
    async def body(conn):
        run_id, fake = await new_run(conn), Fake()
        await advance(conn, run_id, fake.activities, concurrency=1)
        study_id = (await describe(conn, run_id))["study_id"]
        await conn.execute("INSERT INTO theme_runs (study_id, question_number, method, params, embedding_model, n_responses) VALUES ($1, 1, 'kmeans', '{}', 'm', 4)", study_id)
        await retry(conn, run_id, "embed")
        assert await steps(conn, run_id) == {"guide": "done", "review": "done", "respondents": "done", "embed": "pending", "themes": "pending", "synthesize": "pending"}
        assert await conn.fetchval("SELECT count(*) FROM theme_runs WHERE study_id = $1", study_id) == 0  # compensation
        assert (await describe(conn, run_id))["report"] is None
        assert await respondent_count(conn, run_id) == 4
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "done"
        assert fake.calls == {"guide": 1, "review": 0, "respondents": 4, "embed": 2, "themes": 2, "synthesize": 2}

    in_rolled_back_transaction(body)


def test_retry_from_respondents_deletes_them_and_regenerates_under_the_same_study():
    async def body(conn):
        run_id, fake = await new_run(conn), Fake()
        await advance(conn, run_id, fake.activities, concurrency=1)
        study_id = (await describe(conn, run_id))["study_id"]
        await retry(conn, run_id, "respondents")
        assert await respondent_count(conn, run_id) == 0
        await advance(conn, run_id, fake.activities, concurrency=1)
        assert (await describe(conn, run_id))["study_id"] == study_id and await respondent_count(conn, run_id) == 4
        assert fake.calls["guide"] == 1 and fake.calls["respondents"] == 8

    in_rolled_back_transaction(body)


def test_retry_from_guide_removes_the_whole_study_and_makes_a_new_one():
    async def body(conn):
        run_id, fake = await new_run(conn), Fake()
        await advance(conn, run_id, fake.activities, concurrency=1)
        old = (await describe(conn, run_id))["study_id"]
        await retry(conn, run_id, "guide")
        assert await conn.fetchval("SELECT count(*) FROM studies WHERE id = $1", old) == 0
        await advance(conn, run_id, fake.activities, concurrency=1)
        assert (await describe(conn, run_id))["study_id"] not in (None, old)

    in_rolled_back_transaction(body)


def test_retry_continues_after_a_failure_when_no_step_is_named():
    async def body(conn):
        run_id, fake = await new_run(conn), Fake()
        fake.embed_failures = 1
        with pytest.raises(RuntimeError):
            await advance(conn, run_id, fake.activities, concurrency=1)
        await workflow._set_status(conn, run_id, "failed", "dead-lettered")
        await retry(conn, run_id)
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "done" and fake.calls["guide"] == 1

    in_rolled_back_transaction(body)


def test_a_guide_step_that_died_after_creating_the_study_starts_clean_instead_of_leaving_an_orphan():
    async def body(conn):
        run_id, fake = await new_run(conn), Fake()
        orphan = await conn.fetchval("INSERT INTO studies (brief) VALUES ('half-made') RETURNING id")
        await conn.execute("UPDATE workflow_runs SET study_id = $2 WHERE id = $1", run_id, orphan)
        await advance(conn, run_id, fake.activities, concurrency=1)
        assert await conn.fetchval("SELECT count(*) FROM studies WHERE id = $1", orphan) == 0
        assert (await describe(conn, run_id))["study_id"] not in (None, orphan)

    in_rolled_back_transaction(body)


# ---- the event handler, the lock and dead letters --------------------------------------------------------


def test_the_worker_drives_a_run_and_retries_a_failed_step_through_the_event_queue():
    async def body(conn):
        run_id, fake = await new_run(conn), Fake()
        fake.embed_failures = 1
        handlers = {ADVANCE: Handler(lambda c, p: handle_advance(c, p, fake.activities), mark_failed)}
        while await work_once(conn, handlers):
            pass
        assert await run_status(conn, run_id) == "running"  # the step failed, the event waits out its backoff
        await conn.execute("UPDATE events SET available_at = clock_timestamp() WHERE status = 'pending'")
        while await work_once(conn, handlers):
            pass
        assert await run_status(conn, run_id) == "done" and fake.calls["embed"] == 2

    in_rolled_back_transaction(body)


def test_a_dead_lettered_event_marks_a_running_run_failed_but_leaves_a_paused_one_alone():
    async def body(conn):
        running, paused = await new_run(conn), await new_run(conn)
        await workflow._set_status(conn, running, "running")
        await pause(conn, paused)
        await mark_failed(conn, {"run_id": running}, "boom")
        await mark_failed(conn, {"run_id": paused}, "boom")
        assert (await run_status(conn, running), await run_status(conn, paused)) == ("failed", "paused")

    in_rolled_back_transaction(body)


def test_only_one_executor_can_advance_a_run_at_a_time():
    async def run():
        try:
            a, b = await connect(), await connect()
        except (OSError, asyncpg.PostgresError):
            pytest.skip("Postgres is not running (start it with make up)")
        run_id = 2_000_000_001
        try:
            assert await a.fetchval("SELECT pg_try_advisory_lock($1, $2)", workflow.LOCK_NAMESPACE, run_id)  # another executor holds it
            assert await advance(b, run_id) == "busy"
            await a.execute("SELECT pg_advisory_unlock($1, $2)", workflow.LOCK_NAMESPACE, run_id)
            with pytest.raises(WorkflowError, match="No workflow run"):  # free now: b gets the lock and finds no such run
                await advance(b, run_id)
        finally:
            await a.close()
            await b.close()

    asyncio.run(run())


def test_the_api_rejects_runs_too_small_for_the_themes_step():
    from pydantic import ValidationError

    from app.main import WorkflowRequest

    with pytest.raises(ValidationError):
        WorkflowRequest(brief="b", respondents=workflow.MIN_RESPONDENTS - 1)
    assert WorkflowRequest(brief="b", respondents=workflow.MIN_RESPONDENTS).respondents == workflow.MIN_RESPONDENTS


# ---- the human review gate -------------------------------------------------------------------------------------


async def review_run(conn, count=4):
    return await start(conn, "a brief", count, 1, "m", review_guide=True)


def test_a_run_that_asked_for_review_pauses_after_the_guide_and_waits():
    async def body(conn):
        run_id, fake = await review_run(conn), Fake()
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "paused"
        assert await steps(conn, run_id) == {"guide": "done", "review": "pending", "respondents": "pending", "embed": "pending", "themes": "pending", "synthesize": "pending"}
        assert fake.calls["respondents"] == 0
        types = [e["type"] for e in await run_events.events_after(conn, run_id, 0)]
        assert types[-1] == "review.requested"

    in_rolled_back_transaction(body)


def test_approving_with_edits_uses_the_edited_questions_for_every_respondent():
    async def body(conn):
        run_id, fake = await review_run(conn), Fake()
        await advance(conn, run_id, fake.activities, concurrency=1)
        await workflow.approve_guide(conn, run_id, ["  My own   question? ", "", "A second one?"])
        assert await run_status(conn, run_id) == "running"
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "done"
        info = await describe(conn, run_id)
        asked = await conn.fetch("SELECT DISTINCT question_number, question FROM responses s JOIN respondents r ON r.id = s.respondent_id WHERE r.study_id = $1 ORDER BY 1", info["study_id"])
        assert [r["question"] for r in asked] == ["My own question?", "A second one?"]
        assert fake.calls["guide"] == 1  # the guide was not generated again

    in_rolled_back_transaction(body)


def test_approve_is_rejected_unless_the_run_is_waiting_and_the_edit_is_sane():
    async def body(conn):
        run_id, fake = await review_run(conn), Fake()
        with pytest.raises(WorkflowError, match="not waiting"):
            await workflow.approve_guide(conn, run_id, ["q?"])  # nothing generated yet
        await advance(conn, run_id, fake.activities, concurrency=1)
        for bad in ([], ["", "  "], [f"q{i}?" for i in range(11)]):
            with pytest.raises(WorkflowError, match="between 1 and 10"):
                await workflow.approve_guide(conn, run_id, bad)
        await workflow.approve_guide(conn, run_id, ["ok?"])
        with pytest.raises(WorkflowError, match="not waiting"):
            await workflow.approve_guide(conn, run_id, ["again?"])  # already approved
        with pytest.raises(workflow.WorkflowNotFound):
            await workflow.approve_guide(conn, 123456789, ["q?"])

    in_rolled_back_transaction(body)


def test_a_run_without_review_never_stops_for_one():
    async def body(conn):
        run_id, fake = await new_run(conn), Fake()
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "done"
        assert next(s for s in (await describe(conn, run_id))["steps"] if s["name"] == "review")["output"] == {"approved": False}

    in_rolled_back_transaction(body)


def test_retrying_from_the_guide_asks_for_review_again():
    async def body(conn):
        run_id, fake = await review_run(conn), Fake()
        await advance(conn, run_id, fake.activities, concurrency=1)
        await workflow.approve_guide(conn, run_id, ["q1?", "q2?"])
        await advance(conn, run_id, fake.activities, concurrency=1)
        await retry(conn, run_id, "guide")
        assert await advance(conn, run_id, fake.activities, concurrency=1) == "paused"

    in_rolled_back_transaction(body)


# ---- cost tagging and the read side ----------------------------------------------------------------------------


def test_model_calls_made_during_a_run_are_tagged_with_its_id_and_summed():
    async def body(conn):
        from app.db import current_run

        run_id, fake = await new_run(conn), Fake()
        seen = []
        original = fake.guide

        async def spy(brief, model):
            seen.append(current_run.get())
            return await original(brief, model)

        acts = fake.activities
        acts.guide = spy
        await advance(conn, run_id, acts, concurrency=1)
        assert seen == [run_id] and current_run.get() is None  # set while the run works, cleared after
        for cost, tokens in ((0.5, 100), (0.25, 50)):
            await conn.execute("INSERT INTO llm_calls (model, messages, output, input_tokens, output_tokens, latency_ms, cost_usd, run_id) VALUES ('m', '[]', 'o', $1, 0, 1, $2, $3)", tokens, cost, run_id)
        assert await workflow.usage(conn, run_id) == {"cost_usd": 0.75, "tokens": 150, "llm_calls": 2}
        finished = [json.loads(e["data"]) for e in await run_events.events_after(conn, run_id, 0) if e["type"] == "step.finished"]
        assert all("cost_usd" in d for d in finished)

    in_rolled_back_transaction(body)


def test_the_run_list_shows_newest_first_with_cost():
    async def body(conn):
        first, second = await new_run(conn), await new_run(conn)
        await conn.execute("INSERT INTO llm_calls (model, messages, output, input_tokens, output_tokens, latency_ms, cost_usd, run_id) VALUES ('m', '[]', 'o', 1, 1, 1, 0.5, $1)", first)
        runs = await workflow.list_runs(conn, limit=2)
        assert [r["id"] for r in runs] == [second, first]
        assert runs[1]["cost_usd"] == 0.5 and runs[0]["status"] == "pending"

    in_rolled_back_transaction(body)
