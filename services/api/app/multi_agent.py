"""Four roles answer one question together: a planner splits it, analysts research the parts in parallel,
a synthesizer writes the answer and a critic looks for gaps (one extra round at most).

The roles pass typed objects (Plan, Finding, Critique), never each other's chat history. An analyst is the
M7 agent loop run on one sub-question. The final answer is checked by the M8 verifier against every number
and quote the analysts' tools returned.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import asyncpg

from app.agent import AgentRun, build_catalog, run_agent
from app.db import connect
from app.llm.client import complete, complete_streaming, complete_structured
from app.prompts.registry import PromptRef, load
from app.schemas import Critique, Plan
from app.verifier import Verification, verify_answer

PLANNER = ("planner", "v1")
SYNTHESIZER = ("synthesizer", "v1")
CRITIC = ("critic", "v1")
ROLE_MODEL = "gpt-4o-mini"
CONCURRENCY = 3  # analysts running at once: each holds a database connection and an API request
MAX_ROUNDS = 1  # extra analyst rounds the critic may ask for


@dataclass
class Finding:
    sub_question: str
    answer: str | None
    verified: bool | None
    status: str
    run_id: int
    tool_results: list[str] = field(repr=False, default_factory=list)  # what the analyst's tools returned, for verifying the report


@dataclass
class Research:
    question: str
    plan: list[str]
    findings: list[Finding]
    answer: str
    critiques: list[Critique]
    verification: Verification
    seconds: float


ConnectionFactory = Callable[[], AsyncIterator[asyncpg.Connection]]
EventCallback = Callable[[str, str], Awaitable[None]]  # ("delta", text) for each piece of a draft, ("reset", "") when a new draft starts


@asynccontextmanager
async def own_connection() -> AsyncIterator[asyncpg.Connection]:
    conn = await connect()
    try:
        yield conn
    finally:
        await conn.close()


def _finding(sub_question: str, run: AgentRun) -> Finding:
    results = [s.result for s in run.steps if s.kind == "tool" and not s.is_error]
    return Finding(sub_question, run.answer, run.verified, run.status, run.run_id, results)


def _findings_text(findings: list[Finding]) -> str:
    return "\n\n".join(f"Sub-question: {f.sub_question}\nAnswer: {f.answer or '(no answer: ' + f.status + ')'}" for f in findings)


async def plan(conn: asyncpg.Connection, question: str, model: str = ROLE_MODEL) -> Plan:
    messages = [
        {"role": "system", "content": load(*PLANNER, catalog=await build_catalog(conn))},
        {"role": "user", "content": question},
    ]
    return await complete_structured(messages, model, Plan, PromptRef(*PLANNER, {"question": question}))


async def analyze(
    sub_questions: list[str], connection: ConnectionFactory = own_connection, concurrency: int = CONCURRENCY
) -> list[Finding]:
    """Run one analyst per sub-question, at most `concurrency` at a time. The results keep the order of the sub-questions."""
    gate = asyncio.Semaphore(concurrency)

    async def one(sub_question: str) -> Finding:
        async with gate, connection() as conn:
            return _finding(sub_question, await run_agent(conn, sub_question))

    return list(await asyncio.gather(*(one(q) for q in sub_questions)))


async def synthesize(question: str, findings: list[Finding], model: str = ROLE_MODEL, on_event: EventCallback | None = None) -> str:
    messages = [
        {"role": "system", "content": load(*SYNTHESIZER)},
        {"role": "user", "content": f"Question: {question}\n\nFindings:\n\n{_findings_text(findings)}"},
    ]
    prompt = PromptRef(*SYNTHESIZER, {"question": question})
    if on_event is None:
        return (await complete(messages, model, prompt)).text

    async def delta(text: str) -> None:
        await on_event("delta", text)

    return (await complete_streaming(messages, model, delta, prompt)).text


async def critique(question: str, draft: str, findings: list[Finding], model: str = ROLE_MODEL) -> Critique:
    messages = [
        {"role": "system", "content": load(*CRITIC)},
        {"role": "user", "content": f"Question: {question}\n\nDraft:\n{draft}\n\nFindings:\n\n{_findings_text(findings)}"},
    ]
    return await complete_structured(messages, model, Critique, PromptRef(*CRITIC, {"question": question}))


async def research(
    conn: asyncpg.Connection,
    question: str,
    connection: ConnectionFactory = own_connection,
    concurrency: int = CONCURRENCY,
    max_rounds: int = MAX_ROUNDS,
    on_event: EventCallback | None = None,
) -> Research:
    start = time.perf_counter()
    sub_questions = (await plan(conn, question)).sub_questions
    findings = await analyze(sub_questions, connection, concurrency)
    answer = await synthesize(question, findings, on_event=on_event)
    critiques: list[Critique] = []
    for _ in range(max_rounds):
        verdict = await critique(question, answer, findings)
        critiques.append(verdict)
        if verdict.ok:
            break
        findings += await analyze(verdict.gaps, connection, concurrency)
        if on_event is not None:
            await on_event("reset", "")  # the draft already streamed is being replaced
        answer = await synthesize(question, findings, on_event=on_event)
    verification = await verify_answer(conn, answer, [r for f in findings for r in f.tool_results])
    return Research(question, sub_questions, findings, answer, critiques, verification, time.perf_counter() - start)


async def research_question(question: str) -> Research:
    async with own_connection() as conn:
        return await research(conn, question)
