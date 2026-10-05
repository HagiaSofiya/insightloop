"""The agent loop: the model chooses a tool, we run it, add the result to the conversation and repeat.

It stops when the model answers, or at a hard limit on turns or tokens. Every tool call is written to
agent_steps as it happens, so even a run that crashes leaves its trace.
"""

import json
from dataclasses import asdict, dataclass, field

import asyncpg

from app.db import connect
from app.embeddings import EMBEDDING_MODEL
from app.llm.client import complete_with_tools
from app.prompts.registry import PromptRef, load
from app.tools import ToolContext, run_tool, tool_specs
from app.verifier import Problem, Verification, format_verification, verify_answer

PROMPT_NAME = "agent"
PROMPT_VERSION = "v2"
AGENT_MODEL = "gpt-4o-mini"
MAX_STEPS = 6  # model turns
MAX_TOKENS = 40_000  # input plus output tokens across the whole run


@dataclass
class Step:
    number: int
    kind: str  # "tool" or "final"
    tool_name: str | None
    arguments: object  # the parsed arguments, or the raw text when it was not valid JSON
    result: str
    is_error: bool
    input_tokens: int  # the model turn's tokens go on the turn's first step, so the steps add up to the total
    output_tokens: int
    latency_ms: float


@dataclass
class AgentRun:
    run_id: int
    question: str
    model: str
    status: str  # answered, forced_answer (the tools were withdrawn at the step limit), step_limit or token_budget
    answer: str | None
    steps: list[Step] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0
    verified: bool | None = None  # None when verification was switched off
    problems: list[Problem] = field(default_factory=list)
    retried: bool = False  # the first answer failed verification and the agent was asked to fix it
    quotes_checked: int = 0
    numbers_checked: int = 0


async def build_catalog(conn: asyncpg.Connection) -> str:
    """What the agent may look at: each study with its id, size and questions."""
    studies = await conn.fetch(
        """
        SELECT st.id, st.brief, count(DISTINCT r.id) AS respondents, min(r.age) AS lo, max(r.age) AS hi
        FROM studies st JOIN respondents r ON r.study_id = st.id GROUP BY st.id ORDER BY st.id
        """
    )
    questions = await conn.fetch(
        """
        SELECT DISTINCT r.study_id, s.question_number, s.question
        FROM responses s JOIN respondents r ON r.id = s.respondent_id ORDER BY 1, 2
        """
    )
    lines = []
    for study in studies:
        ages = f", ages {study['lo']} to {study['hi']}" if study["lo"] is not None else ""
        lines.append(f"Study {study['id']}: {study['brief']} ({study['respondents']} respondents{ages})")
        lines += [f"  Q{q['question_number']}. {q['question']}" for q in questions if q["study_id"] == study["id"]]
    return "\n".join(lines) or "(no studies yet)"


async def _insert_step(conn: asyncpg.Connection, run_id: int, step: Step) -> None:
    await conn.execute(
        """
        INSERT INTO agent_steps (run_id, step_number, kind, tool_name, arguments, result, is_error,
                                 input_tokens, output_tokens, latency_ms)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7, $8, $9, $10)
        """,
        run_id, step.number, step.kind, step.tool_name, json.dumps(step.arguments), step.result,
        step.is_error, step.input_tokens, step.output_tokens, step.latency_ms,
    )


def _retry_message(verification: Verification) -> str:
    listed = "\n".join(f"- {p.kind}: {p.claim}: {p.detail}" for p in verification.problems)
    return (
        "Your answer failed an automatic check against the data:\n"
        f"{listed}\n\n"
        "Fix these. Quote only exact words from tool results, put the right response id after each quote, "
        "and state only counts and percentages that appear in tool results (call count_by_theme if you need "
        "them). Then give your corrected answer."
    )


async def run_agent(
    conn: asyncpg.Connection,
    question: str,
    model: str = AGENT_MODEL,
    max_steps: int = MAX_STEPS,
    max_tokens: int = MAX_TOKENS,
    embedding_model: str = EMBEDDING_MODEL,
    verify: bool = True,
) -> AgentRun:
    catalog = await build_catalog(conn)
    messages: list[dict] = [
        {"role": "system", "content": load(PROMPT_NAME, PROMPT_VERSION, catalog=catalog, max_steps=max_steps)},
        {"role": "user", "content": question},
    ]
    prompt = PromptRef(PROMPT_NAME, PROMPT_VERSION, {"question": question, "max_steps": max_steps})
    specs, ctx = tool_specs(), ToolContext(conn, embedding_model)

    run_id = await conn.fetchval(
        "INSERT INTO agent_runs (question, model, status, max_steps, max_tokens) VALUES ($1, $2, 'running', $3, $4) RETURNING id",
        question, model, max_steps, max_tokens,
    )
    run = AgentRun(run_id, question, model, "step_limit", None)
    step_number = 0

    for turn_number in range(1, max_steps + 1):
        if run.input_tokens + run.output_tokens >= max_tokens:
            run.status = "token_budget"
            break
        forced = turn_number == max_steps  # the last allowed turn: take the tools away so it must answer
        turn = await complete_with_tools(
            list(messages), model, specs, "none" if forced else "auto", prompt
        )
        run.input_tokens += turn.input_tokens
        run.output_tokens += turn.output_tokens
        run.cost_usd = None if run.cost_usd is None or turn.cost_usd is None else run.cost_usd + turn.cost_usd

        if not turn.tool_calls:
            step_number += 1
            answer = turn.content or ""
            final = Step(step_number, "final", None, {}, answer, False, turn.input_tokens, turn.output_tokens, turn.latency_ms)
            run.steps.append(final)
            await _insert_step(conn, run_id, final)

            verification = None
            if verify:
                results = [s.result for s in run.steps if s.kind == "tool" and not s.is_error]
                verification = await verify_answer(conn, answer, results)
                can_retry = (
                    not verification.verified and not run.retried and not forced
                    and run.input_tokens + run.output_tokens < max_tokens
                )
                if can_retry:  # one chance to fix it, with the problems listed
                    run.retried = True
                    messages.append({"role": "assistant", "content": answer})
                    messages.append({"role": "user", "content": _retry_message(verification)})
                    continue
            run.answer = answer
            run.status = "forced_answer" if forced else "answered"
            if verification is not None:
                run.verified, run.problems = verification.verified, verification.problems
                run.quotes_checked, run.numbers_checked = verification.quotes_checked, verification.numbers_checked
            break

        messages.append(turn.assistant_message())
        for index, call in enumerate(turn.tool_calls):
            result, is_error = await run_tool(call.name, call.arguments, ctx)
            try:
                arguments = json.loads(call.arguments or "{}")
            except json.JSONDecodeError:
                arguments = call.arguments  # keep what the model actually wrote
            step_number += 1
            first = index == 0
            step = Step(
                step_number, "tool", call.name, arguments, result, is_error,
                turn.input_tokens if first else 0, turn.output_tokens if first else 0, turn.latency_ms if first else 0.0,
            )
            run.steps.append(step)
            await _insert_step(conn, run_id, step)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    await conn.execute(
        """
        UPDATE agent_runs SET status = $2, answer = $3, input_tokens = $4, output_tokens = $5, cost_usd = $6,
                              verified = $7, problems = $8::jsonb, retried = $9
        WHERE id = $1
        """,
        run_id, run.status, run.answer, run.input_tokens, run.output_tokens, run.cost_usd,
        run.verified, json.dumps([asdict(p) for p in run.problems]), run.retried,
    )
    return run


async def agent_answer(question: str, max_steps: int = MAX_STEPS, max_tokens: int = MAX_TOKENS) -> AgentRun:
    conn = await connect()
    try:
        return await run_agent(conn, question, max_steps=max_steps, max_tokens=max_tokens)
    finally:
        await conn.close()


def _preview(result: str, limit: int = 110) -> str:
    text = " ".join(result.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_agent_run(run: AgentRun) -> str:
    lines = [f"question: {run.question}", ""]
    for step in run.steps:
        if step.kind == "tool":
            flag = "  ERROR" if step.is_error else ""
            lines.append(f"{step.number}. {step.tool_name} {json.dumps(step.arguments, ensure_ascii=False)}{flag}")
            lines.append(f"     -> {_preview(step.result)}")
    lines += ["", f"answer ({run.status}):", run.answer if run.answer else "(no answer: the run was stopped by a limit)"]
    if run.verified is not None:
        retry = "after one retry, " if run.retried else ""
        lines.append(retry + format_verification(Verification(run.verified, run.problems, run.quotes_checked, run.numbers_checked)))
    cost = f"${run.cost_usd:.4f}" if run.cost_usd is not None else "unknown cost"
    tool_steps = sum(step.kind == "tool" for step in run.steps)
    lines += ["", f"{tool_steps} tool calls, {run.input_tokens:,} tokens in and {run.output_tokens:,} out, {cost} (run {run.run_id})"]
    return "\n".join(lines)
