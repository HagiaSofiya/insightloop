"""A study run as a durable state machine: guide, respondents, embed, themes, synthesize.

Every step's state is a row in workflow_steps, so a run that crashes resumes at the first step that is not done
(the checkpoint). The code that decides what runs next (the orchestration) is kept apart from the code that does
the work (the activities), so tests can swap the activities for fakes and never touch a model.

A run advances when a `workflow.advance` event is handled (see worker.py). Two things make that safe:
the event queue redelivers it if the worker dies, and an advisory lock stops two executors from advancing the
same run at once. Pause, cancel and retry are status changes the executor notices between steps and before
each respondent.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import asyncpg

from app.db import current_run
from app.embeddings import embed_responses
from app.events import enqueue
from app.golden import ensure_themes
from app.llm.client import complete_structured
from app.multi_agent import EventCallback, research
from app.personas import generate_persona, sample_profiles
from app.prompts.registry import PromptRef, build_messages
from app.run_events import emit
from app.respondents import interview
from app.schemas import DiscussionGuide

logger = logging.getLogger(__name__)

ADVANCE = "workflow.advance"
STEPS = ["guide", "review", "respondents", "embed", "themes", "synthesize"]
MAX_QUESTIONS = 4
CONCURRENCY = 5  # respondents generated at once
MIN_RESPONDENTS = 10  # the themes step needs 8 answers per question and the respondents step may lose 20%
MIN_SUCCESS = 0.8  # the respondents step passes if at least this share of respondents completed
GUIDE_VERSION = "v2"
LOCK_NAMESPACE = 7012  # the first half of the advisory lock key; the run id is the second half


class WorkflowError(Exception):
    """The request does not make sense for this run's state (the API turns it into a 409)."""


class WorkflowNotFound(WorkflowError):
    """There is no run with that id (the API turns it into a 404)."""


class Interrupted(Exception):
    """The run was paused or cancelled while a step was working. The step stops at a safe point."""


class StepFailed(Exception):
    """A step could not finish. Its progress is kept, and a retry continues from it."""


# ---- the work: swapped for fakes in tests ----------------------------------------------------------


@dataclass
class Activities:
    guide: Callable[[str, str], Awaitable[list[str]]]  # (brief, model) -> questions
    respondent: Callable[[str, dict, list[str], str], Awaitable[tuple[dict, list[str]]]]  # -> (persona fields, answers)
    embed: Callable[[asyncpg.Connection, int], Awaitable[None]]  # (conn, study_id)
    themes: Callable[[asyncpg.Connection, int, int], Awaitable[None]]  # (conn, study_id, number of questions)
    synthesize: Callable[[asyncpg.Connection, int, EventCallback], Awaitable[tuple[str, dict]]]  # -> (report, details for the reasoning panel)


async def _guide(brief: str, model: str) -> list[str]:
    inputs = {"brief": brief}
    guide = await complete_structured(
        build_messages("discussion_guide", GUIDE_VERSION, inputs), model, DiscussionGuide,
        PromptRef("discussion_guide", GUIDE_VERSION, inputs),
    )
    return [q.text for q in guide.questions[:MAX_QUESTIONS]]


async def _respondent(brief: str, profile: dict, questions: list[str], model: str) -> tuple[dict, list[str]]:
    persona = await generate_persona(brief, profile, model)
    answers = await interview(persona, questions, model, asyncio.Semaphore(1))
    return persona.model_dump(), answers


async def _embed(conn: asyncpg.Connection, study_id: int) -> None:
    await embed_responses(conn, study_id=study_id)


async def _themes(conn: asyncpg.Connection, study_id: int, questions: int) -> None:
    await ensure_themes(conn, study_id, questions)


async def _synthesize(conn: asyncpg.Connection, study_id: int, on_event: EventCallback) -> tuple[str, dict]:
    question = f"In study {study_id}: what are the main findings of this study, and what should the team do about them?"
    result = await research(conn, question, on_event=on_event)
    details = {
        "plan": result.plan,
        "analysts": [{"run_id": f.run_id, "sub_question": f.sub_question, "status": f.status, "verified": f.verified} for f in result.findings],
        "critiques": [{"ok": c.ok, "gaps": c.gaps} for c in result.critiques],
        "verification": {
            "verified": result.verification.verified, "quotes_checked": result.verification.quotes_checked,
            "numbers_checked": result.verification.numbers_checked, "problems": result.verification.as_json(),
        },
    }
    return result.answer, details


REAL = Activities(_guide, _respondent, _embed, _themes, _synthesize)


# ---- starting and controlling a run -----------------------------------------------------------------


async def start(conn: asyncpg.Connection, brief: str, respondent_count: int, seed: int, model: str, review_guide: bool = False) -> int:
    """Create the run, its step rows and the event that starts it, in one transaction (the outbox again)."""
    async with conn.transaction():
        run_id = await conn.fetchval(
            "INSERT INTO workflow_runs (brief, respondent_count, seed, model, review_guide) VALUES ($1, $2, $3, $4, $5) RETURNING id",
            brief, respondent_count, seed, model, review_guide,
        )
        await conn.executemany(
            "INSERT INTO workflow_steps (run_id, name, position) VALUES ($1, $2, $3)",
            [(run_id, name, position) for position, name in enumerate(STEPS)],
        )
        await enqueue(conn, ADVANCE, {"run_id": run_id})
        await emit(conn, run_id, "run.created", {"respondents": respondent_count, "steps": STEPS, "review_guide": review_guide})
    return run_id


async def _status(conn: asyncpg.Connection, run_id: int) -> str:
    status = await conn.fetchval("SELECT status FROM workflow_runs WHERE id = $1", run_id)
    if status is None:
        raise WorkflowNotFound(f"No workflow run with id {run_id}.")
    return status


async def _set_status(conn: asyncpg.Connection, run_id: int, status: str, error: str | None = None) -> None:
    await conn.execute("UPDATE workflow_runs SET status = $2, error = $3, updated_at = now() WHERE id = $1", run_id, status, error)


async def pause(conn: asyncpg.Connection, run_id: int) -> None:
    """Ask the run to stop at the next safe point. A run that is not running or pending cannot be paused."""
    async with conn.transaction():
        changed = await conn.fetchval(
            "UPDATE workflow_runs SET status = 'paused', updated_at = now() WHERE id = $1 AND status IN ('pending', 'running') RETURNING id",
            run_id,
        )
        if changed is None:
            raise WorkflowError(f"Run {run_id} is {await _status(conn, run_id)}, so it cannot be paused.")
        await emit(conn, run_id, "run.paused")


async def resume(conn: asyncpg.Connection, run_id: int) -> None:
    async with conn.transaction():
        changed = await conn.fetchval(
            "UPDATE workflow_runs SET status = 'running', updated_at = now() WHERE id = $1 AND status = 'paused' RETURNING id", run_id
        )
        if changed is None:
            raise WorkflowError(f"Run {run_id} is {await _status(conn, run_id)}, so it cannot be resumed.")
        await enqueue(conn, ADVANCE, {"run_id": run_id})
        await emit(conn, run_id, "run.resumed")


async def cancel(conn: asyncpg.Connection, run_id: int) -> None:
    """Stop for good. Work already done is kept (the data so far stays in the study)."""
    async with conn.transaction():
        changed = await conn.fetchval(
            "UPDATE workflow_runs SET status = 'cancelled', updated_at = now() "
            "WHERE id = $1 AND status IN ('pending', 'running', 'paused', 'failed') RETURNING id",
            run_id,
        )
        if changed is None:
            raise WorkflowError(f"Run {run_id} is {await _status(conn, run_id)}, so it cannot be cancelled.")
        await emit(conn, run_id, "run.cancelled")


MAX_GUIDE_QUESTIONS = 10


async def approve_guide(conn: asyncpg.Connection, run_id: int, questions: list[str]) -> None:
    """The human's edit: replace the guide's questions, approve the guide and let the run continue."""
    cleaned = [" ".join(q.split()) for q in questions if q.strip()]
    if not 1 <= len(cleaned) <= MAX_GUIDE_QUESTIONS:
        raise WorkflowError(f"A guide needs between 1 and {MAX_GUIDE_QUESTIONS} questions, got {len(cleaned)}.")
    async with conn.transaction():
        run = await conn.fetchrow("SELECT status, review_guide, guide_approved FROM workflow_runs WHERE id = $1 FOR UPDATE", run_id)
        if run is None:
            raise WorkflowNotFound(f"No workflow run with id {run_id}.")
        guide = await conn.fetchval("SELECT output FROM workflow_steps WHERE run_id = $1 AND name = 'guide' AND status = 'done'", run_id)
        if not (run["status"] == "paused" and run["review_guide"] and not run["guide_approved"] and guide):
            raise WorkflowError(f"Run {run_id} is not waiting for a guide review (status {run['status']}).")
        output = {**json.loads(guide), "questions": cleaned}
        await conn.execute("UPDATE workflow_steps SET output = $2::jsonb WHERE run_id = $1 AND name = 'guide'", run_id, json.dumps(output))
        await conn.execute("UPDATE workflow_runs SET guide_approved = true, status = 'running', updated_at = now() WHERE id = $1", run_id)
        await enqueue(conn, ADVANCE, {"run_id": run_id})
        await emit(conn, run_id, "guide.approved", {"questions": cleaned})


async def retry(conn: asyncpg.Connection, run_id: int, from_step: str | None = None) -> None:
    """Run again. With no from_step, continue after the failure. With one, undo that step and every later step
    (latest first) and run from there."""
    if from_step is not None and from_step not in STEPS:
        raise WorkflowError(f"Unknown step {from_step!r}. The steps are {STEPS}.")
    async with conn.transaction():
        run = await conn.fetchrow("SELECT status, study_id FROM workflow_runs WHERE id = $1 FOR UPDATE", run_id)
        if run is None:
            raise WorkflowNotFound(f"No workflow run with id {run_id}.")
        if run["status"] not in ("failed", "done", "paused"):
            raise WorkflowError(f"Run {run_id} is {run['status']}, so it cannot be retried. Only failed, paused or finished runs can.")
        if from_step is not None:
            for name in reversed(STEPS[STEPS.index(from_step):]):
                await _undo(conn, run_id, name, run["study_id"])
        await conn.execute("UPDATE workflow_runs SET status = 'running', error = NULL, updated_at = now() WHERE id = $1", run_id)
        await enqueue(conn, ADVANCE, {"run_id": run_id})
        await emit(conn, run_id, "run.retried", {"from_step": from_step})


async def _undo(conn: asyncpg.Connection, run_id: int, step: str, study_id: int | None) -> None:
    if study_id is not None:
        if step == "guide":
            await conn.execute("DELETE FROM studies WHERE id = $1", study_id)  # cascades to everything in the study
            await conn.execute("UPDATE workflow_runs SET study_id = NULL, guide_approved = false WHERE id = $1", run_id)
        elif step == "respondents":
            await conn.execute("DELETE FROM respondents WHERE study_id = $1", study_id)
        elif step == "embed":
            await conn.execute(
                "UPDATE responses SET embedding = NULL, embedding_model = NULL "
                "WHERE respondent_id IN (SELECT id FROM respondents WHERE study_id = $1)", study_id,
            )
        elif step == "themes":
            await conn.execute("DELETE FROM theme_runs WHERE study_id = $1", study_id)
    if step == "review":
        await conn.execute("UPDATE workflow_runs SET guide_approved = false WHERE id = $1", run_id)
    if step == "synthesize":
        await conn.execute("UPDATE workflow_runs SET report = NULL WHERE id = $1", run_id)
    await conn.execute(
        "UPDATE workflow_steps SET status = 'pending', output = NULL, error = NULL, started_at = NULL, finished_at = NULL "
        "WHERE run_id = $1 AND name = $2",
        run_id, step,
    )


# ---- the executor -------------------------------------------------------------------------------------


@dataclass
class Ctx:
    conn: asyncpg.Connection
    run_id: int
    acts: Activities
    concurrency: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # one connection serves every task, so take turns


async def _check(ctx: Ctx) -> None:
    """Stop if the run was paused, cancelled or retried out from under us. Called at every safe point."""
    async with ctx.lock:
        status = await ctx.conn.fetchval("SELECT status FROM workflow_runs WHERE id = $1", ctx.run_id)
    if status != "running":
        raise Interrupted(status)


async def _run(ctx: Ctx) -> asyncpg.Record:
    return await ctx.conn.fetchrow("SELECT * FROM workflow_runs WHERE id = $1", ctx.run_id)


async def _output(ctx: Ctx, step: str) -> dict:
    return json.loads(await ctx.conn.fetchval("SELECT output FROM workflow_steps WHERE run_id = $1 AND name = $2", ctx.run_id, step) or "{}")


async def step_guide(ctx: Ctx) -> dict:
    run = await _run(ctx)
    if run["study_id"] is not None:  # an earlier attempt made a study but died before recording the step: start clean
        await _undo(ctx.conn, ctx.run_id, "guide", run["study_id"])
        await ctx.conn.execute("UPDATE workflow_steps SET status = 'running' WHERE run_id = $1 AND name = 'guide'", ctx.run_id)
    questions = await ctx.acts.guide(run["brief"], run["model"])
    async with ctx.conn.transaction():
        study_id = await ctx.conn.fetchval(
            "INSERT INTO studies (brief, seed, model) VALUES ($1, $2, $3) RETURNING id", run["brief"], run["seed"], run["model"]
        )
        await ctx.conn.execute("UPDATE workflow_runs SET study_id = $2, updated_at = now() WHERE id = $1", ctx.run_id, study_id)
    return {"study_id": study_id, "questions": questions}


async def step_review(ctx: Ctx) -> dict:
    """The human gate: if the run asked for a review and nobody approved the guide yet, pause until they do."""
    run = await _run(ctx)
    if not run["review_guide"] or run["guide_approved"]:
        return {"approved": bool(run["guide_approved"])}
    questions = (await _output(ctx, "guide"))["questions"]
    async with ctx.conn.transaction():
        await ctx.conn.execute("UPDATE workflow_runs SET status = 'paused', updated_at = now() WHERE id = $1 AND status = 'running'", ctx.run_id)
        await emit(ctx.conn, ctx.run_id, "review.requested", {"questions": questions})
    raise Interrupted("paused")


async def step_respondents(ctx: Ctx) -> dict:
    run, guide = await _run(ctx), await _output(ctx, "guide")
    study_id, questions, count = run["study_id"], guide["questions"], run["respondent_count"]
    profiles = sample_profiles(count, run["seed"])
    existing = {row["slot"] for row in await ctx.conn.fetch("SELECT slot FROM respondents WHERE study_id = $1 AND slot IS NOT NULL", study_id)}
    gate = asyncio.Semaphore(ctx.concurrency)

    async def one(slot: int) -> None:
        async with gate:
            await _check(ctx)
            persona, answers = await ctx.acts.respondent(run["brief"], profiles[slot], questions, run["model"])
            async with ctx.lock, ctx.conn.transaction():  # the respondent and all their answers, or nothing
                respondent_id = await ctx.conn.fetchval(
                    """
                    INSERT INTO respondents (study_id, source, slot, name, age, gender, city, income, household, occupation, backstory, speaking_style)
                    VALUES ($1, 'synthetic', $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (study_id, slot) WHERE slot IS NOT NULL DO NOTHING
                    RETURNING id
                    """,
                    study_id, slot, *(persona.get(k) for k in ("name", "age", "gender", "city", "income", "household", "occupation", "backstory", "speaking_style")),
                )
                if respondent_id is not None:  # None means another delivery already saved this slot
                    await ctx.conn.executemany(
                        "INSERT INTO responses (respondent_id, question_number, question, answer) VALUES ($1, $2, $3, $4)",
                        [(respondent_id, n, q, a) for n, (q, a) in enumerate(zip(questions, answers, strict=True), 1)],
                    )
                    saved = await ctx.conn.fetchval("SELECT count(*) FROM respondents WHERE study_id = $1 AND slot IS NOT NULL", study_id)
                    await emit(ctx.conn, ctx.run_id, "respondent.saved", {"slot": slot, "completed": saved, "total": count})

    missing = [slot for slot in range(count) if slot not in existing]
    results = await asyncio.gather(*(one(slot) for slot in missing), return_exceptions=True)
    stopped = next((r for r in results if isinstance(r, Interrupted)), None)
    if stopped is not None:
        raise stopped  # work in flight finished and was saved; the rest is picked up on resume
    failures = {slot: r for slot, r in zip(missing, results) if isinstance(r, BaseException)}
    done = await ctx.conn.fetchval("SELECT count(*) FROM respondents WHERE study_id = $1 AND slot IS NOT NULL", study_id)
    if done < MIN_SUCCESS * count:
        first = next(iter(failures.values()), None)
        raise StepFailed(f"only {done} of {count} respondents completed" + (f": {type(first).__name__}: {first}" if first else ""))
    return {"completed": done, "failed_slots": sorted(failures), "errors": [f"{type(e).__name__}: {e}"[:200] for e in list(failures.values())[:3]]}


async def step_embed(ctx: Ctx) -> dict:
    await ctx.acts.embed(ctx.conn, (await _run(ctx))["study_id"])
    return {}


async def step_themes(ctx: Ctx) -> dict:
    questions = (await _output(ctx, "guide"))["questions"]
    await ctx.acts.themes(ctx.conn, (await _run(ctx))["study_id"], len(questions))
    return {"questions": len(questions)}


class TokenBatcher:
    """Groups the model's tiny pieces into events: one per ~60 characters or 0.3 seconds, not one per token."""

    def __init__(self, flush: Callable[[str], Awaitable[None]], min_chars: int = 60, max_seconds: float = 0.3, clock=time.monotonic):
        self.flush, self.min_chars, self.max_seconds, self.clock = flush, min_chars, max_seconds, clock
        self.buffer, self.last = "", clock()

    async def add(self, text: str) -> None:
        self.buffer += text
        if len(self.buffer) >= self.min_chars or self.clock() - self.last >= self.max_seconds:
            await self.finish()

    async def finish(self) -> None:
        if self.buffer:
            text, self.buffer = self.buffer, ""
            await self.flush(text)
        self.last = self.clock()


async def step_synthesize(ctx: Ctx) -> dict:
    async def flush(text: str) -> None:
        await emit(ctx.conn, ctx.run_id, "report.delta", {"text": text})

    batcher = TokenBatcher(flush)

    async def on_event(kind: str, text: str) -> None:
        if kind == "delta":
            await batcher.add(text)
        else:  # a new draft replaces the one streamed so far
            await batcher.finish()
            await emit(ctx.conn, ctx.run_id, "report.reset")

    await emit(ctx.conn, ctx.run_id, "report.reset")  # a retried step starts from a clean slate in the client too
    report, details = await ctx.acts.synthesize(ctx.conn, (await _run(ctx))["study_id"], on_event)
    await batcher.finish()
    await ctx.conn.execute("UPDATE workflow_runs SET report = $2, updated_at = now() WHERE id = $1", ctx.run_id, report)
    return {"chars": len(report), **details}


STEP_FUNCTIONS = {
    "guide": step_guide, "review": step_review, "respondents": step_respondents, "embed": step_embed, "themes": step_themes, "synthesize": step_synthesize,
}


async def _run_step(ctx: Ctx, name: str) -> None:
    async with ctx.conn.transaction():
        attempts = await ctx.conn.fetchval(
            "UPDATE workflow_steps SET status = 'running', attempts = attempts + 1, started_at = now(), error = NULL "
            "WHERE run_id = $1 AND name = $2 RETURNING attempts",
            ctx.run_id, name,
        )
        await emit(ctx.conn, ctx.run_id, "step.started", {"step": name, "attempt": attempts})
    try:
        output = await STEP_FUNCTIONS[name](ctx)
    except Interrupted:
        await ctx.conn.execute("UPDATE workflow_steps SET status = 'pending' WHERE run_id = $1 AND name = $2 AND status = 'running'", ctx.run_id, name)
        raise
    except Exception as error:
        message = f"{type(error).__name__}: {error}"[:2000]
        async with ctx.conn.transaction():
            await ctx.conn.execute("UPDATE workflow_steps SET status = 'failed', error = $3 WHERE run_id = $1 AND name = $2", ctx.run_id, name, message)
            await emit(ctx.conn, ctx.run_id, "step.failed", {"step": name, "error": message})
        raise
    async with ctx.conn.transaction():
        await ctx.conn.execute(
            "UPDATE workflow_steps SET status = 'done', output = $3::jsonb, finished_at = now() WHERE run_id = $1 AND name = $2 AND status = 'running'",
            ctx.run_id, name, json.dumps(output),
        )
        await emit(ctx.conn, ctx.run_id, "step.finished", {"step": name, "output": output, **await usage(ctx.conn, ctx.run_id)})


async def usage(conn: asyncpg.Connection, run_id: int) -> dict:
    """What the run has spent so far, from the call ledger."""
    row = await conn.fetchrow(
        "SELECT coalesce(sum(cost_usd), 0) AS cost, coalesce(sum(input_tokens + output_tokens), 0) AS tokens, count(*) AS calls "
        "FROM llm_calls WHERE run_id = $1", run_id,
    )
    return {"cost_usd": float(row["cost"]), "tokens": int(row["tokens"]), "llm_calls": int(row["calls"])}


async def advance(conn: asyncpg.Connection, run_id: int, acts: Activities = REAL, concurrency: int = CONCURRENCY) -> str:
    """Run steps until the run is done, paused, cancelled or a step fails. Returns the run's status ("busy" if another
    executor already holds this run). A failing step raises, so the event queue retries it later."""
    if not await conn.fetchval("SELECT pg_try_advisory_lock($1, $2)", LOCK_NAMESPACE, run_id):
        logger.info("workflow %s is already being advanced elsewhere", run_id)
        return "busy"
    token = current_run.set(run_id)  # from here on, every model call is logged against this run
    try:
        status = await _status(conn, run_id)
        if status == "pending":
            await _set_status(conn, run_id, "running")
        elif status != "running":
            return status
        ctx = Ctx(conn, run_id, acts, concurrency)
        try:
            while True:
                await _check(ctx)
                step = await conn.fetchval("SELECT name FROM workflow_steps WHERE run_id = $1 AND status <> 'done' ORDER BY position LIMIT 1", run_id)
                if step is None:
                    async with conn.transaction():
                        await _set_status(conn, run_id, "done")
                        await emit(conn, run_id, "run.done")
                    return "done"
                logger.info("workflow %s: running step %s", run_id, step)
                await _run_step(ctx, step)
        except Interrupted as stop:
            logger.info("workflow %s stopped: %s", run_id, stop)
            return str(stop)
    finally:
        current_run.reset(token)
        await conn.execute("SELECT pg_advisory_unlock($1, $2)", LOCK_NAMESPACE, run_id)


async def handle_advance(conn: asyncpg.Connection, payload: dict, acts: Activities = REAL) -> None:
    await advance(conn, payload["run_id"], acts)


async def mark_failed(conn: asyncpg.Connection, payload: dict, error: str) -> None:
    """The event was dead-lettered: show the run as failed (not stuck on running)."""
    async with conn.transaction():
        changed = await conn.fetchval(
            "UPDATE workflow_runs SET status = 'failed', error = $2, updated_at = now() WHERE id = $1 AND status IN ('pending', 'running') RETURNING id",
            payload["run_id"], error[:2000],
        )
        if changed is not None:
            await emit(conn, payload["run_id"], "run.failed", {"error": error[:500]})


async def describe(conn: asyncpg.Connection, run_id: int) -> dict | None:
    run = await conn.fetchrow("SELECT * FROM workflow_runs WHERE id = $1", run_id)
    if run is None:
        return None
    steps = await conn.fetch("SELECT name, status, attempts, output, error FROM workflow_steps WHERE run_id = $1 ORDER BY position", run_id)
    return {
        **{k: run[k] for k in ("id", "brief", "respondent_count", "status", "study_id", "report", "error", "review_guide", "guide_approved")},
        "created_at": run["created_at"].isoformat(),
        **await usage(conn, run_id),
        "steps": [{**{k: s[k] for k in ("name", "status", "attempts", "error")}, "output": json.loads(s["output"]) if s["output"] else None} for s in steps],
    }


async def list_runs(conn: asyncpg.Connection, limit: int = 50) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT w.id, w.brief, w.respondent_count, w.status, w.study_id, w.created_at,
               coalesce(sum(c.cost_usd), 0) AS cost
        FROM workflow_runs w LEFT JOIN llm_calls c ON c.run_id = w.id
        GROUP BY w.id ORDER BY w.id DESC LIMIT $1
        """,
        limit,
    )
    return [{**{k: r[k] for k in ("id", "brief", "respondent_count", "status", "study_id")}, "created_at": r["created_at"].isoformat(), "cost_usd": float(r["cost"])} for r in rows]
