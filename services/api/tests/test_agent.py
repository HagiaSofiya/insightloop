import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from app import cli, tools
from app.agent import build_catalog, format_agent_run, run_agent
from app.llm.client import ToolCall, ToolTurn, complete_with_tools
from app.tools import tool_specs
from dbtools import call, in_rolled_back_transaction, script, simulation
from app.store import insert_simulation


async def seed(conn):
    study_id = await insert_simulation(conn, "a meal-kit study", 0, "m", simulation())
    ids = {
        row["answer"]: row["id"]
        for row in await conn.fetch(
            "SELECT s.id, s.answer FROM responses s JOIN respondents r ON r.id = s.respondent_id WHERE r.study_id = $1",
            study_id,
        )
    }
    return study_id, ids


async def steps_in_db(conn, run_id):
    return await conn.fetch("SELECT * FROM agent_steps WHERE run_id = $1 ORDER BY step_number", run_id)


# ---- the loop ----------------------------------------------------------------------------------


def test_the_model_calls_tools_then_answers_and_every_step_is_stored(stub_llm, ledger):
    async def body(conn):
        study_id, ids = await seed(conn)
        script(stub_llm, [
            [call("get_quotes", {"response_ids": [ids["a1"]]})],
            [call("get_quotes", {"response_ids": [ids["b2"], ids["c1"]]})],
            'People said "a1" [R1].',
        ])

        run = await run_agent(conn, "Why did they stop?", "m", embedding_model="m")

        assert (run.status, run.answer) == ("answered", 'People said "a1" [R1].')
        assert [(s.number, s.kind, s.tool_name) for s in run.steps] == [
            (1, "tool", "get_quotes"), (2, "tool", "get_quotes"), (3, "final", None),
        ]
        assert (run.input_tokens, run.output_tokens) == (300, 60)  # three model turns of 100 and 20

        stored = await steps_in_db(conn, run.run_id)
        assert [(r["step_number"], r["kind"], r["tool_name"], r["is_error"]) for r in stored] == [
            (1, "tool", "get_quotes", False), (2, "tool", "get_quotes", False), (3, "final", None, False),
        ]
        assert json.loads(stored[0]["arguments"]) == {"response_ids": [ids["a1"]]}
        assert json.loads(stored[0]["result"])["quotes"][0]["answer"] == "a1"
        assert stored[2]["result"] == 'People said "a1" [R1].'
        assert sum(r["input_tokens"] for r in stored) == 300 and sum(r["output_tokens"] for r in stored) == 60

        saved = await conn.fetchrow("SELECT * FROM agent_runs WHERE id = $1", run.run_id)
        assert (saved["status"], saved["answer"], saved["question"]) == ("answered", run.answer, "Why did they stop?")
        assert (saved["input_tokens"], saved["output_tokens"], saved["max_steps"], saved["max_tokens"]) == (300, 60, 6, 40000)

    in_rolled_back_transaction(body)


def test_each_tool_result_goes_back_to_the_model_in_the_shape_the_api_expects(stub_llm):
    async def body(conn):
        _, ids = await seed(conn)
        script(stub_llm, [[call("get_quotes", {"response_ids": [ids["a1"]]})], "done"])

        await run_agent(conn, "q", "m", embedding_model="m")

        second_call = stub_llm.calls[1]
        assert [m["role"] for m in second_call] == ["system", "user", "assistant", "tool"]
        asked, answered = second_call[2], second_call[3]
        assert asked["content"] is None and asked["tool_calls"][0]["type"] == "function"
        assert asked["tool_calls"][0]["function"] == {"name": "get_quotes", "arguments": json.dumps({"response_ids": [ids["a1"]]})}
        assert answered["tool_call_id"] == asked["tool_calls"][0]["id"]  # the result is tied to the call it answers
        assert json.loads(answered["content"])["quotes"][0]["answer"] == "a1"
        assert stub_llm.options[0]["tool_choice"] == "auto"
        assert [t["function"]["name"] for t in stub_llm.options[0]["tools"]] == [t["function"]["name"] for t in tool_specs()]

    in_rolled_back_transaction(body)


def test_the_system_prompt_lists_the_studies_and_their_questions(stub_llm):
    async def body(conn):
        study_id, _ = await seed(conn)
        script(stub_llm, ["answer"])

        await run_agent(conn, "q", "m", max_steps=4, embedding_model="m")

        system = stub_llm.calls[0][0]["content"]
        assert f"Study {study_id}: a meal-kit study (3 respondents, ages 25 to 61)" in system
        assert "  Q1. Why try it?" in system and "  Q2. Why stop?" in system
        assert "at most 4 model turns" in system

    in_rolled_back_transaction(body)


def test_the_catalog_has_a_line_per_study_then_its_questions():
    async def body(conn):
        study_id, _ = await seed(conn)

        catalog = await build_catalog(conn)

        assert f"Study {study_id}: a meal-kit study (3 respondents, ages 25 to 61)\n  Q1. Why try it?\n  Q2. Why stop?" in catalog

    in_rolled_back_transaction(body)


# ---- errors are fed back ------------------------------------------------------------------------


def test_a_tool_error_goes_back_to_the_model_which_then_corrects_itself(stub_llm):
    async def body(conn):
        _, ids = await seed(conn)
        script(stub_llm, [
            [call("get_quotes", {"response_ids": [999999999]})],  # an id that does not exist
            [call("get_quotes", {"response_ids": [ids["a1"]]})],  # the corrected call
            "Fixed it.",
        ])

        run = await run_agent(conn, "q", "m", embedding_model="m")

        feedback = json.loads(stub_llm.calls[1][3]["content"])
        assert "No responses with ids [999999999]" in feedback["error"]  # the model was told exactly what was wrong
        assert [s.is_error for s in run.steps] == [True, False, False]
        assert run.status == "answered"
        stored = await steps_in_db(conn, run.run_id)
        assert [r["is_error"] for r in stored] == [True, False, False]

    in_rolled_back_transaction(body)


def test_unknown_tools_and_broken_arguments_are_fed_back_and_kept_as_written(stub_llm):
    async def body(conn):
        script(stub_llm, [
            [call("fly_to_the_moon", {})],
            [call("get_quotes", "{this is not json")],
            "gave up",
        ])

        run = await run_agent(conn, "q", "m", embedding_model="m")

        assert "Unknown tool 'fly_to_the_moon'" in json.loads(stub_llm.calls[1][3]["content"])["error"]
        assert "not valid JSON" in json.loads(stub_llm.calls[2][5]["content"])["error"]
        assert [s.is_error for s in run.steps[:2]] == [True, True]
        stored = await steps_in_db(conn, run.run_id)
        assert json.loads(stored[1]["arguments"]) == "{this is not json"  # the raw text, so the trace shows what the model wrote

    in_rolled_back_transaction(body)


def test_parallel_tool_calls_in_one_turn_all_run_and_count_tokens_once(stub_llm):
    async def body(conn):
        _, ids = await seed(conn)
        script(stub_llm, [
            [call("get_quotes", {"response_ids": [ids["a1"]]}), call("get_quotes", {"response_ids": [ids["b1"]]})],
            "both read",
        ])

        run = await run_agent(conn, "q", "m", embedding_model="m")

        assert [s.number for s in run.steps] == [1, 2, 3]
        assert [s.input_tokens for s in run.steps[:2]] == [100, 0]  # the turn's tokens sit on its first step only
        assert run.input_tokens == 200 == sum(s.input_tokens for s in run.steps)
        tool_messages = [m for m in stub_llm.calls[1] if m["role"] == "tool"]
        assert len(tool_messages) == 2 and tool_messages[0]["tool_call_id"] != tool_messages[1]["tool_call_id"]

    in_rolled_back_transaction(body)


# ---- the limits ---------------------------------------------------------------------------------


def test_at_the_step_limit_the_tools_are_taken_away_so_the_model_must_answer(stub_llm):
    async def body(conn):
        _, ids = await seed(conn)
        keep_calling = [call("get_quotes", {"response_ids": [ids["a1"]]})]
        script(stub_llm, [keep_calling, keep_calling, "Here is what I found so far."])

        run = await run_agent(conn, "q", "m", max_steps=3, embedding_model="m")

        assert len(stub_llm.calls) == 3  # never more model turns than max_steps
        assert [o["tool_choice"] for o in stub_llm.options] == ["auto", "auto", "none"]
        assert (run.status, run.answer) == ("forced_answer", "Here is what I found so far.")

    in_rolled_back_transaction(body)


def test_a_model_that_ignores_the_withdrawn_tools_ends_with_no_answer(stub_llm):
    async def body(conn):
        _, ids = await seed(conn)
        keep_calling = [call("get_quotes", {"response_ids": [ids["a1"]]})]
        script(stub_llm, [keep_calling, keep_calling])

        run = await run_agent(conn, "q", "m", max_steps=2, embedding_model="m")

        assert len(stub_llm.calls) == 2 and run.status == "step_limit" and run.answer is None
        saved = await conn.fetchrow("SELECT status, answer FROM agent_runs WHERE id = $1", run.run_id)
        assert (saved["status"], saved["answer"]) == ("step_limit", None)

    in_rolled_back_transaction(body)


def test_the_token_budget_stops_the_run_before_the_next_model_call(stub_llm):
    async def body(conn):
        _, ids = await seed(conn)
        keep_calling = [call("get_quotes", {"response_ids": [ids["a1"]]})]
        script(stub_llm, [keep_calling, keep_calling, keep_calling], tokens=(500, 100))

        run = await run_agent(conn, "q", "m", max_tokens=1000, embedding_model="m")

        assert len(stub_llm.calls) == 2  # 600 after the first turn, 1200 after the second, so a third is not allowed
        assert run.status == "token_budget" and run.answer is None
        assert run.input_tokens + run.output_tokens == 1200

    in_rolled_back_transaction(body)


def test_a_real_bug_stops_the_run_but_the_trace_so_far_survives(stub_llm, monkeypatch):
    async def body(conn):
        _, ids = await seed(conn)
        script(stub_llm, [
            [call("get_quotes", {"response_ids": [ids["a1"]]})],
            [call("get_quotes", {"response_ids": [ids["a1"]]})],
        ])
        calls_made = {"n": 0}
        real = tools.TOOLS["get_quotes"].handler

        async def breaks_second_time(ctx, args):
            calls_made["n"] += 1
            if calls_made["n"] == 2:
                raise RuntimeError("a bug in our code")
            return await real(ctx, args)

        monkeypatch.setattr(tools.TOOLS["get_quotes"], "handler", breaks_second_time)

        with pytest.raises(RuntimeError, match="a bug in our code"):
            await run_agent(conn, "q", "m", embedding_model="m")

        rows = await conn.fetch("SELECT * FROM agent_steps ORDER BY id DESC LIMIT 1")
        saved = await conn.fetchrow("SELECT status FROM agent_runs ORDER BY id DESC LIMIT 1")
        assert rows[0]["step_number"] == 1 and saved["status"] == "running"  # the first step was written before the crash

    in_rolled_back_transaction(body)


# ---- the adapter and the report -----------------------------------------------------------------


def test_the_tool_turn_adapter_reads_calls_and_records_the_turn(stub_llm, ledger):
    script(stub_llm, [[call("get_quotes", {"response_ids": [1]}), call("get_theme", {"study_id": 1, "question_number": 1})]])

    turn = asyncio.run(complete_with_tools([{"role": "user", "content": "q"}], "m", tool_specs()))

    assert [(c.name, json.loads(c.arguments)) for c in turn.tool_calls] == [
        ("get_quotes", {"response_ids": [1]}), ("get_theme", {"study_id": 1, "question_number": 1}),
    ]
    assert (turn.input_tokens, turn.output_tokens, turn.content) == (100, 20, None)
    assert ledger[0]["params"] == {"tools": ["search_responses", "get_theme", "count_by_theme", "get_quotes"], "tool_choice": "auto"}
    assert json.loads(ledger[0]["output"])[0]["name"] == "get_quotes"  # no text, so the calls are what was recorded


def test_assistant_message_has_tool_calls_only_when_there_are_some():
    with_calls = ToolTurn(None, [ToolCall("c1", "get_quotes", "{}")], 1, 1, 1.0, None)
    plain = ToolTurn("hello", [], 1, 1, 1.0, None)

    assert with_calls.assistant_message() == {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_quotes", "arguments": "{}"}}],
    }
    assert plain.assistant_message() == {"role": "assistant", "content": "hello"}


def test_a_stored_tool_turn_cannot_be_replayed(monkeypatch):
    async def fake_fetch_call(call_id):
        return {
            "id": call_id, "messages": "[]", "params": json.dumps({"tools": ["get_quotes"], "tool_choice": "auto"}),
            "prompt_name": None, "model": "m", "output": "",
        }

    monkeypatch.setattr(cli, "fetch_call", fake_fetch_call)

    with pytest.raises(cli.NotReplayable, match="used tools .get_quotes."):
        asyncio.run(cli.replay(5))


def test_the_report_shows_each_tool_call_flags_errors_and_the_totals(stub_llm):
    async def body(conn):
        _, ids = await seed(conn)
        script(stub_llm, [
            [call("get_quotes", {"response_ids": [999999999]})],
            [call("get_quotes", {"response_ids": [ids["a1"]]})],
            "Final words.",
        ])
        run = await run_agent(conn, "Why did they stop?", "m", embedding_model="m")

        text = format_agent_run(run)

        assert text.startswith("question: Why did they stop?")
        assert '1. get_quotes {"response_ids": [999999999]}  ERROR' in text
        assert f'2. get_quotes {{"response_ids": [{ids["a1"]}]}}\n' in text
        assert "answer (answered):\nFinal words." in text
        assert "2 tool calls, 300 tokens in and 60 out" in text and f"(run {run.run_id})" in text

    in_rolled_back_transaction(body)


# ---- verification and the single retry ----------------------------------------------------------

REAL = "the price is too high for what you get"
INVENTED = "the price is outrageous and everyone hates it"


async def seed_with_wording(conn):
    study_id, ids = await seed(conn)
    await conn.execute("UPDATE responses SET answer = $1 WHERE id = $2", f"Honestly, {REAL}. I stopped.", ids["a1"])
    return study_id, ids


def test_a_failed_answer_gets_one_retry_with_the_problems_listed(stub_llm):
    async def body(conn):
        _, ids = await seed_with_wording(conn)
        read_it = [call("get_quotes", {"response_ids": [ids["a1"]]})]
        script(stub_llm, [
            read_it,
            f'They said "{INVENTED}" [R{ids["a1"]}].',  # the model invents a quote
            f'They said "{REAL}" [R{ids["a1"]}].',  # the corrected answer
        ])

        run = await run_agent(conn, "q", "m", embedding_model="m")

        assert run.verified is True and run.retried is True and run.problems == []
        assert run.answer == f'They said "{REAL}" [R{ids["a1"]}].'
        assert len(stub_llm.calls) == 3
        history = stub_llm.calls[2]
        assert history[-2] == {"role": "assistant", "content": f'They said "{INVENTED}" [R{ids["a1"]}].'}  # the bad answer
        assert history[-1]["role"] == "user" and "failed an automatic check" in history[-1]["content"]
        assert INVENTED in history[-1]["content"] and "appear in no response" in history[-1]["content"]  # told exactly why
        assert [s.kind for s in run.steps] == ["tool", "final", "final"]  # both attempts are in the trace

        saved = await conn.fetchrow("SELECT status, answer, verified, retried, problems FROM agent_runs WHERE id = $1", run.run_id)
        assert (saved["status"], saved["answer"], saved["verified"], saved["retried"]) == ("answered", run.answer, True, True)
        assert json.loads(saved["problems"]) == []

    in_rolled_back_transaction(body)


def test_an_answer_that_fails_again_is_marked_unverified_with_the_reasons(stub_llm):
    async def body(conn):
        _, ids = await seed_with_wording(conn)
        script(stub_llm, [
            [call("get_quotes", {"response_ids": [ids["a1"]]})],
            f'They said "{INVENTED}" [R{ids["a1"]}].',
            f'They said "{INVENTED} again" [R{ids["a1"]}].',  # still wrong after the retry
        ])

        run = await run_agent(conn, "q", "m", embedding_model="m")

        assert len(stub_llm.calls) == 3  # one retry and no more
        assert (run.status, run.verified, run.retried) == ("answered", False, True)
        assert run.answer.endswith(f'again" [R{ids["a1"]}].')  # the last answer is the one that is kept, flagged
        assert [p.kind for p in run.problems] == ["quote"] and "appear in no response" in run.problems[0].detail
        saved = await conn.fetchrow("SELECT verified, problems FROM agent_runs WHERE id = $1", run.run_id)
        assert saved["verified"] is False and json.loads(saved["problems"])[0]["kind"] == "quote"

    in_rolled_back_transaction(body)


def test_a_verified_answer_is_not_retried(stub_llm):
    async def body(conn):
        _, ids = await seed_with_wording(conn)
        script(stub_llm, [
            [call("get_quotes", {"response_ids": [ids["a1"]]})],
            f'They said "{REAL}" [R{ids["a1"]}].',
        ])

        run = await run_agent(conn, "q", "m", embedding_model="m")

        assert len(stub_llm.calls) == 2 and run.verified is True and run.retried is False

    in_rolled_back_transaction(body)


def test_there_is_no_retry_on_the_last_allowed_turn(stub_llm):
    async def body(conn):
        _, ids = await seed_with_wording(conn)
        script(stub_llm, [f'They said "{INVENTED}" [R{ids["a1"]}].'])

        run = await run_agent(conn, "q", "m", max_steps=1, embedding_model="m")  # the only turn is the forced one

        assert len(stub_llm.calls) == 1 and run.verified is False and run.retried is False

    in_rolled_back_transaction(body)


def test_there_is_no_retry_when_the_token_budget_is_spent(stub_llm):
    async def body(conn):
        _, ids = await seed_with_wording(conn)
        script(stub_llm, [f'They said "{INVENTED}" [R{ids["a1"]}].'], tokens=(1500, 0))

        run = await run_agent(conn, "q", "m", max_tokens=1000, embedding_model="m")

        assert len(stub_llm.calls) == 1 and run.verified is False and run.retried is False and run.answer

    in_rolled_back_transaction(body)


def test_verification_can_be_switched_off(stub_llm):
    async def body(conn):
        _, ids = await seed_with_wording(conn)
        script(stub_llm, [f'They said "{INVENTED}" [R{ids["a1"]}].'])

        run = await run_agent(conn, "q", "m", embedding_model="m", verify=False)

        assert run.verified is None and run.retried is False and len(stub_llm.calls) == 1

    in_rolled_back_transaction(body)


def test_a_wrong_number_is_caught_against_the_tool_results(stub_llm):
    async def body(conn):
        _, ids = await seed_with_wording(conn)
        script(stub_llm, [
            [call("get_quotes", {"response_ids": [ids["a1"]]})],
            "47 respondents hated the price.",  # no tool result contains 47
            "The data I read does not give a count.",
        ])

        run = await run_agent(conn, "q", "m", embedding_model="m")

        assert run.retried and run.verified is True  # the second answer makes no unsupported claim
        assert "no tool result contains the number 47" in stub_llm.calls[2][-1]["content"]

    in_rolled_back_transaction(body)


def test_the_report_shows_the_verification_outcome(stub_llm):
    async def body(conn):
        _, ids = await seed_with_wording(conn)
        script(stub_llm, [
            [call("get_quotes", {"response_ids": [ids["a1"]]})], f'"{INVENTED}" [R{ids["a1"]}]', f'"{INVENTED} too" [R{ids["a1"]}]',
        ])
        bad = format_agent_run(await run_agent(conn, "q", "m", embedding_model="m"))
        script(stub_llm, [[call("get_quotes", {"response_ids": [ids["a1"]]})], f'"{REAL}" [R{ids["a1"]}]'])
        good = format_agent_run(await run_agent(conn, "q", "m", embedding_model="m"))

        assert "after one retry, UNVERIFIED (1 quotes and 0 numbers checked):" in bad and "appear in no response" in bad
        assert "\nverified (1 quotes and 0 numbers checked)" in good and "UNVERIFIED" not in good

    in_rolled_back_transaction(body)
