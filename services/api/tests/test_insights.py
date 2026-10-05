import json

from app.insights import agent_run, source_response, study_insights
from app.store import insert_simulation
from dbtools import in_rolled_back_transaction, simulation


async def seed_themes(conn):
    study_id = await insert_simulation(conn, "a meal-kit study", 0, "m", simulation())
    rows = await conn.fetch(
        "SELECT s.id, s.answer FROM responses s JOIN respondents r ON r.id = s.respondent_id WHERE r.study_id = $1 AND s.question_number = 1 ORDER BY s.id", study_id
    )
    run_id = await conn.fetchval(
        "INSERT INTO theme_runs (study_id, question_number, method, params, embedding_model, n_responses, silhouette) VALUES ($1, 1, 'kmeans', $2::jsonb, 'm', 3, 0.2) RETURNING id",
        study_id, json.dumps({"question": "Why try it?"}),
    )
    big = await conn.fetchval("INSERT INTO themes (run_id, kind, label, description, size) VALUES ($1, 'theme', 'Big', 'the big one', 2) RETURNING id", run_id)
    out = await conn.fetchval("INSERT INTO themes (run_id, kind, label, description, size) VALUES ($1, 'outlier', 'Outliers', 'rest', 1) RETURNING id", run_id)
    await conn.executemany("INSERT INTO theme_members (theme_id, response_id, distance) VALUES ($1, $2, $3)", [(big, rows[0]["id"], 0.2), (big, rows[1]["id"], 0.1), (out, rows[2]["id"], 0.5)])
    return study_id, rows


def test_insights_list_themes_with_shares_and_closest_examples_and_outliers_last():
    async def body(conn):
        study_id, rows = await seed_themes(conn)
        result = await study_insights(conn, study_id)
        [question] = result["questions"]
        assert (question["question_number"], question["question"], question["responses"]) == (1, "Why try it?", 3)
        big, outliers = question["themes"]
        assert (big["label"], big["size"], big["share"], big["kind"]) == ("Big", 2, 0.667, "theme")
        assert [e["response_id"] for e in big["examples"]] == [rows[1]["id"], rows[0]["id"]]  # closest to the center first
        assert outliers["kind"] == "outlier" and outliers["share"] == 0.333
        assert await study_insights(conn, 987654321) is None

    in_rolled_back_transaction(body)


def test_a_source_response_comes_with_its_question_and_respondent():
    async def body(conn):
        study_id, rows = await seed_themes(conn)
        source = await source_response(conn, rows[0]["id"])
        assert (source["answer"], source["question"], source["name"], source["age"]) == ("a1", "Why try it?", "Ann", 25)
        assert await source_response(conn, 987654321) is None

    in_rolled_back_transaction(body)


def test_an_agent_run_shows_its_steps_and_verifier_outcome():
    async def body(conn):
        run_id = await conn.fetchval(
            "INSERT INTO agent_runs (question, model, status, answer, max_steps, max_tokens, cost_usd, verified, problems) VALUES ('q', 'm', 'answered', 'a', 6, 100, 0.001, false, $1::jsonb) RETURNING id",
            json.dumps([{"kind": "number", "claim": "40 respondents", "detail": "no tool result contains the number 40"}]),
        )
        await conn.execute(
            "INSERT INTO agent_steps (run_id, step_number, kind, tool_name, arguments, result, input_tokens, output_tokens, latency_ms) VALUES ($1, 1, 'tool', 'get_quotes', '{\"response_ids\": [1]}', $2, 10, 5, 12.4)",
            run_id, "x " * 2000,
        )
        run = await agent_run(conn, run_id)
        assert run["verified"] is False and run["problems"][0]["kind"] == "number" and run["cost_usd"] == 0.001
        [step] = run["steps"]
        assert (step["tool"], step["arguments"], step["tokens"], step["latency_ms"]) == ("get_quotes", {"response_ids": [1]}, 15, 12)
        assert len(step["result"]) <= 1500 and step["result"].endswith("...")
        assert await agent_run(conn, 987654321) is None

    in_rolled_back_transaction(body)
