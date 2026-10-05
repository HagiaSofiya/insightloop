import json

import pytest

from app import tools
from app.store import insert_simulation
from app.tools import (
    CountArgs, GetThemeArgs, QuotesArgs, SearchArgs, ToolContext, ToolError, count_by_theme, get_quotes,
    get_theme, run_tool, search_responses, tool_specs,
)
from dbtools import in_rolled_back_transaction, place, script_question, simulation

# a1 and a2 are answers by the 25 year old, b1 and b2 by the 40 year old, c1 and c2 by the 61 year old
DISTANCES = {"a1": 0.10, "a2": 0.30, "b1": 0.45, "b2": 0.55, "c1": 0.70, "c2": 0.90}


async def seed(conn):
    study_id = await insert_simulation(conn, "a meal-kit study", 0, "m", simulation())
    ids = await place(conn, study_id, DISTANCES)
    return study_id, ids


async def seed_themes(conn, study_id, ids):
    """One theme run for question 1: 'Price worries' holds the 25 and 40 year olds, 'Fit' the 61 year old."""
    run_id = await conn.fetchval(
        "INSERT INTO theme_runs (study_id, question_number, method, embedding_model, n_responses) "
        "VALUES ($1, 1, 'kmeans', 'm', 3) RETURNING id",
        study_id,
    )
    for label, description, size, members in [
        ("Price worries", "They find it too expensive.", 2, [(ids["a1"], 0.05), (ids["b1"], 0.2)]),
        ("Fit problems", "It did not fit their routine.", 1, [(ids["c1"], 0.1)]),
    ]:
        theme_id = await conn.fetchval(
            "INSERT INTO themes (run_id, kind, label, description, size) VALUES ($1, 'theme', $2, $3, $4) RETURNING id",
            run_id, label, description, size,
        )
        for response_id, distance in members:
            await conn.execute("INSERT INTO theme_members VALUES ($1, $2, $3)", theme_id, response_id, distance)
    return run_id


# ---- search_responses --------------------------------------------------------------------------


def test_search_returns_the_closest_responses_with_a_match_strength(stub_llm):
    script_question(stub_llm)

    async def body(conn):
        study_id, ids = await seed(conn)

        result = await search_responses(ToolContext(conn, "m"), SearchArgs(query="anything", study_id=study_id, limit=6))

        by_id = {r["response_id"]: r for r in result["results"]}
        assert result["count"] == 6 and result["results"][0]["response_id"] == ids["a1"]
        assert by_id[ids["a2"]]["match"] == "strong" and by_id[ids["b1"]]["match"] == "strong"  # 0.30 and 0.45 are within 0.50
        assert by_id[ids["b2"]]["match"] == "weak" and by_id[ids["c2"]]["match"] == "weak"  # 0.55 and 0.90 are not
        assert result["results"][0] == {
            "response_id": ids["a1"], "study_id": study_id, "question_number": 1, "age": 25, "gender": "woman",
            "match": "strong", "answer": "a1",
        }

    in_rolled_back_transaction(body)


def test_search_filters_by_age_and_gender_and_limit(stub_llm):
    script_question(stub_llm)

    async def body(conn):
        study_id, ids = await seed(conn)
        ctx = ToolContext(conn, "m")

        older = await search_responses(ctx, SearchArgs(query="x", study_id=study_id, min_age=30, limit=10))
        younger = await search_responses(ctx, SearchArgs(query="x", study_id=study_id, max_age=30, limit=10))
        between = await search_responses(ctx, SearchArgs(query="x", study_id=study_id, min_age=30, max_age=50, limit=10))
        two = await search_responses(ctx, SearchArgs(query="x", study_id=study_id, limit=2))
        nobody = await search_responses(ctx, SearchArgs(query="x", study_id=study_id, gender="man"))

        assert {r["age"] for r in older["results"]} == {40, 61}
        assert {r["age"] for r in younger["results"]} == {25} and younger["count"] == 2
        assert {r["age"] for r in between["results"]} == {40}
        assert two["count"] == 2
        assert nobody["count"] == 0 and nobody["note"] == "No responses matched these filters."

    in_rolled_back_transaction(body)


def test_search_rejects_requests_the_model_can_fix(stub_llm):
    script_question(stub_llm)

    async def body(conn):
        study_id, _ = await seed(conn)
        ctx = ToolContext(conn, "m")

        with pytest.raises(ToolError, match=rf"no study with id 999999.*Study ids: \[.*{study_id}"):
            await search_responses(ctx, SearchArgs(query="x", study_id=999999))
        with pytest.raises(ToolError, match=r"min_age \(50\) is greater than max_age \(30\)"):
            await search_responses(ctx, SearchArgs(query="x", min_age=50, max_age=30))

    in_rolled_back_transaction(body)


# ---- get_theme and count_by_theme --------------------------------------------------------------


def test_get_theme_lists_the_themes_of_a_question():
    async def body(conn):
        study_id, ids = await seed(conn)
        await seed_themes(conn, study_id, ids)

        result = await get_theme(ToolContext(conn, "m"), GetThemeArgs(study_id=study_id, question_number=1))

        assert result["responses"] == 3
        assert result["themes"] == [
            {"label": "Price worries", "description": "They find it too expensive.", "size": 2, "share": 0.67},
            {"label": "Fit problems", "description": "It did not fit their routine.", "size": 1, "share": 0.33},
        ]

    in_rolled_back_transaction(body)


def test_get_theme_shows_one_theme_with_its_most_typical_answers_first():
    async def body(conn):
        study_id, ids = await seed(conn)
        await seed_themes(conn, study_id, ids)

        result = await get_theme(ToolContext(conn, "m"), GetThemeArgs(study_id=study_id, question_number=1, theme="PRICE"))

        assert result["theme"]["label"] == "Price worries"  # a partial, case-insensitive match
        assert [e["response_id"] for e in result["examples"]] == [ids["a1"], ids["b1"]]  # smaller distance first
        assert result["examples"][0]["age"] == 25

    in_rolled_back_transaction(body)


def test_get_theme_errors_tell_the_model_what_exists():
    async def body(conn):
        study_id, ids = await seed(conn)
        await seed_themes(conn, study_id, ids)
        ctx = ToolContext(conn, "m")

        with pytest.raises(ToolError, match=r"labels are: \['Price worries', 'Fit problems'\]"):
            await get_theme(ctx, GetThemeArgs(study_id=study_id, question_number=1, theme="delivery"))
        with pytest.raises(ToolError, match=rf"No themes exist for study {study_id} question 2.*study {study_id}: questions \[1\]"):
            await get_theme(ctx, GetThemeArgs(study_id=study_id, question_number=2))

    in_rolled_back_transaction(body)


def test_counts_by_theme_split_by_age_band_gender_or_nothing():
    async def body(conn):
        study_id, ids = await seed(conn)
        await seed_themes(conn, study_id, ids)
        ctx = ToolContext(conn, "m")

        by_age = await count_by_theme(ctx, CountArgs(study_id=study_id, question_number=1))
        by_gender = await count_by_theme(ctx, CountArgs(study_id=study_id, question_number=1, group_by="gender"))
        overall = await count_by_theme(ctx, CountArgs(study_id=study_id, question_number=1, group_by="none"))

        assert by_age["themes"] == [
            {"theme": "Price worries", "size": 2, "counts": {"18-29": 1, "30-44": 1}},
            {"theme": "Fit problems", "size": 1, "counts": {"60+": 1}},
        ]
        assert by_age["totals"] == {"18-29": 1, "30-44": 1, "60+": 1} and by_age["responses"] == 3
        assert by_gender["totals"] == {"woman": 3}
        assert overall["totals"] == {"all": 3} and overall["themes"][0]["counts"] == {"all": 2}

    in_rolled_back_transaction(body)


# ---- get_quotes --------------------------------------------------------------------------------


def test_get_quotes_returns_full_text_in_the_requested_order():
    async def body(conn):
        study_id, ids = await seed(conn)
        await conn.execute("UPDATE responses SET answer = $1 WHERE id = $2", "x" * 700, ids["c2"])

        result = await get_quotes(ToolContext(conn, "m"), QuotesArgs(response_ids=[ids["c2"], ids["a1"]]))

        assert [q["response_id"] for q in result["quotes"]] == [ids["c2"], ids["a1"]]
        assert len(result["quotes"][0]["answer"]) == 600 and result["quotes"][0]["answer"].endswith("...")  # shortened
        assert result["quotes"][1] == {
            "response_id": ids["a1"], "study_id": study_id, "question_number": 1, "question": "Why try it?",
            "age": 25, "gender": "woman", "answer": "a1",
        }

    in_rolled_back_transaction(body)


def test_get_quotes_names_the_ids_that_do_not_exist():
    async def body(conn):
        _, ids = await seed(conn)

        with pytest.raises(ToolError, match=rf"No responses with ids \[999999999\]"):
            await get_quotes(ToolContext(conn, "m"), QuotesArgs(response_ids=[ids["a1"], 999999999]))

    in_rolled_back_transaction(body)


# ---- running a tool from the model's JSON ------------------------------------------------------


def test_the_model_is_told_about_bad_calls_instead_of_the_run_crashing():
    async def body(conn):
        ctx = ToolContext(conn, "m")
        cases = {
            ("nonexistent_tool", "{}"): "Unknown tool 'nonexistent_tool'. The available tools are:",
            ("get_quotes", "{not json"): "The arguments are not valid JSON",
            ("get_quotes", "[1, 2]"): "The arguments must be a JSON object",
            ("get_quotes", "{}"): "Invalid arguments: response_ids: Field required",
            ("get_quotes", '{"response_ids": [1, 2, 3, 4, 5, 6]}'): "response_ids: List should have at most 5 items",
            ("search_responses", '{"query": "x", "limit": 99}'): "limit: Input should be less than or equal to 10",
            ("search_responses", '{"query": "x", "gender": "robot"}'): "gender: Input should be",
        }
        for (name, arguments), message in cases.items():
            result, is_error = await run_tool(name, arguments, ctx)
            assert is_error is True and message in json.loads(result)["error"], (name, arguments, result)

    in_rolled_back_transaction(body)


def test_a_result_that_is_too_large_is_refused_with_advice(monkeypatch):
    monkeypatch.setattr(tools, "MAX_RESULT_CHARS", 100)

    async def body(conn):
        study_id, ids = await seed(conn)

        result, is_error = await run_tool("get_quotes", json.dumps({"response_ids": list(ids.values())[:5]}), ToolContext(conn, "m"))

        assert is_error and "too large" in json.loads(result)["error"] and "Ask for less" in json.loads(result)["error"]

    in_rolled_back_transaction(body)


def test_a_successful_call_returns_json_and_is_not_an_error():
    async def body(conn):
        _, ids = await seed(conn)

        result, is_error = await run_tool("get_quotes", json.dumps({"response_ids": [ids["a1"]]}), ToolContext(conn, "m"))

        assert is_error is False and json.loads(result)["quotes"][0]["answer"] == "a1"

    in_rolled_back_transaction(body)


def test_a_real_bug_in_a_tool_is_not_hidden_from_us(monkeypatch):
    async def broken(ctx, args):
        raise RuntimeError("a bug in our code")

    monkeypatch.setattr(tools.TOOLS["get_quotes"], "handler", broken)

    async def body(conn):
        with pytest.raises(RuntimeError, match="a bug in our code"):
            await run_tool("get_quotes", '{"response_ids": [1]}', ToolContext(conn, "m"))

    in_rolled_back_transaction(body)


def test_the_tool_specs_describe_every_tool_and_its_arguments():
    specs = tool_specs()

    assert [s["function"]["name"] for s in specs] == ["search_responses", "get_theme", "count_by_theme", "get_quotes"]
    for spec in specs:
        assert spec["type"] == "function" and spec["function"]["description"]
        assert spec["function"]["parameters"]["type"] == "object"
    search = specs[0]["function"]["parameters"]
    assert search["required"] == ["query"] and "price too high" in search["properties"]["query"]["description"]
    assert specs[3]["function"]["parameters"]["properties"]["response_ids"]["maxItems"] == 5
