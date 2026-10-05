import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

from app import multi_agent
from app.multi_agent import analyze, research
from app.schemas import Critique, Plan
from app.store import insert_simulation
from dbtools import call, in_rolled_back_transaction, script, simulation


def same_connection(conn):
    """Every analyst shares the test's one transaction, so they must run one at a time (concurrency=1)."""

    @asynccontextmanager
    async def factory():
        yield conn

    return factory


async def seed(conn):
    study_id = await insert_simulation(conn, "a meal-kit study", 0, "m", simulation())
    rows = await conn.fetch(
        "SELECT s.id, s.answer FROM responses s JOIN respondents r ON r.id = s.respondent_id WHERE r.study_id = $1", study_id
    )
    return {row["answer"]: row["id"] for row in rows}


def go(stub_llm, turns, outcomes, run=lambda conn, **kw: research(conn, "Why did they stop?", **kw), **kwargs):
    stub_llm.outcomes = list(outcomes)
    script(stub_llm, turns)
    result = {}

    async def body(conn):
        ids = await seed(conn)
        result["ids"] = ids
        result["value"] = await research(conn, "Why did they stop?", connection=same_connection(conn), concurrency=1, **kwargs)

    in_rolled_back_transaction(body)
    return result["value"]


# ---- the roles ---------------------------------------------------------------------------------


def test_planner_then_analysts_then_synthesizer_then_critic(stub_llm):
    value = go(
        stub_llm,
        turns=["analyst one says A", "analyst two says B", "The report: A and B."],
        outcomes=[Plan(sub_questions=["part 1?", "part 2?"]), Critique(ok=True)],
    )
    assert value.plan == ["part 1?", "part 2?"]
    assert [(f.sub_question, f.answer) for f in value.findings] == [("part 1?", "analyst one says A"), ("part 2?", "analyst two says B")]
    assert value.answer == "The report: A and B."
    assert [c.ok for c in value.critiques] == [True]


def test_the_synthesizer_sees_the_findings_and_not_the_analysts_conversations(stub_llm):
    go(
        stub_llm,
        turns=["finding A", "finding B", "report"],
        outcomes=[Plan(sub_questions=["part 1?", "part 2?"]), Critique(ok=True)],
    )
    synth_messages = stub_llm.calls[3]  # planner, analyst, analyst, synthesizer
    assert [m["role"] for m in synth_messages] == ["system", "user"]
    assert "finding A" in synth_messages[1]["content"] and "finding B" in synth_messages[1]["content"]
    assert "part 1?" in synth_messages[1]["content"]
    assert not any(m["role"] == "tool" for m in synth_messages)


def test_a_critic_gap_sends_one_more_analyst_and_a_new_synthesis(stub_llm):
    value = go(
        stub_llm,
        turns=["finding A", "draft report", "finding for the gap", "final report"],
        outcomes=[Plan(sub_questions=["part 1?"]), Critique(ok=False, gaps=["what about the gap?"])],
    )
    assert [f.sub_question for f in value.findings] == ["part 1?", "what about the gap?"]
    assert value.answer == "final report"
    assert len(value.critiques) == 1  # one extra round at most: the critic is not asked again


def test_no_critic_runs_when_extra_rounds_are_switched_off(stub_llm):
    value = go(stub_llm, turns=["finding A", "report"], outcomes=[Plan(sub_questions=["part 1?"])], max_rounds=0)
    assert value.critiques == [] and value.answer == "report"


def test_the_report_is_verified_against_what_the_analysts_tools_returned(stub_llm):
    stub_llm.outcomes = [Plan(sub_questions=["part 1?"]), Critique(ok=True)]
    result = {}

    async def body(conn):
        ids = await seed(conn)
        script(stub_llm, [
            [call("get_quotes", {"response_ids": [ids["a1"]]})],
            'One said "a1" [R1].',
            "Forty respondents agreed, which is 40 respondents in all.",
        ])
        result["value"] = await research(conn, "Why?", connection=same_connection(conn), concurrency=1)

    in_rolled_back_transaction(body)
    verification = result["value"].verification
    assert not verification.verified
    assert [p.kind for p in verification.problems] == ["number"]


# ---- fan-out -----------------------------------------------------------------------------------


def test_analysts_run_in_parallel_up_to_the_cap_and_keep_their_order(monkeypatch):
    active, peak = 0, 0

    async def fake_run_agent(conn, question):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01 * (5 - int(question[-1])))  # later questions finish first
        active -= 1
        return NS(run_id=int(question[-1]), answer=f"answer {question}", verified=True, status="answered", steps=[])

    @asynccontextmanager
    async def no_connection():
        yield None

    monkeypatch.setattr(multi_agent, "run_agent", fake_run_agent)
    findings = asyncio.run(analyze([f"q{i}" for i in range(1, 5)], no_connection, concurrency=2))
    assert [f.sub_question for f in findings] == ["q1", "q2", "q3", "q4"]
    assert peak == 2


# ---- the schemas the model must fill -------------------------------------------------------------


def test_a_plan_has_one_to_four_sub_questions():
    with pytest.raises(ValidationError):
        Plan(sub_questions=[])
    with pytest.raises(ValidationError):
        Plan(sub_questions=["a", "b", "c", "d", "e"])


def test_a_critique_must_agree_with_itself():
    with pytest.raises(ValidationError):
        Critique(ok=True, gaps=["something"])
    with pytest.raises(ValidationError):
        Critique(ok=False, gaps=[])
    assert Critique(ok=False, gaps=["what about X?"]).gaps == ["what about X?"]
