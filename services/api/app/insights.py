"""Read-side queries for the UI: a study's themes with example quotes, one source response, one agent run."""

import json

import asyncpg

EXAMPLES = 3  # closest answers shown under each theme
ANSWER_CHARS = 400
RESULT_CHARS = 1500  # per tool result in the reasoning panel


def _short(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


async def study_insights(conn: asyncpg.Connection, study_id: int) -> dict | None:
    study = await conn.fetchrow("SELECT id, brief FROM studies WHERE id = $1", study_id)
    if study is None:
        return None
    runs = await conn.fetch(
        "SELECT DISTINCT ON (question_number) id, question_number, n_responses, silhouette, params "
        "FROM theme_runs WHERE study_id = $1 AND question_number IS NOT NULL ORDER BY question_number, id DESC",
        study_id,
    )
    questions = []
    for run in runs:
        themes = await conn.fetch(
            "SELECT id, kind, label, description, size FROM themes WHERE run_id = $1 ORDER BY (kind = 'outlier'), size DESC, id", run["id"]
        )
        entries = []
        for theme in themes:
            examples = await conn.fetch(
                """
                SELECT s.id AS response_id, r.age, r.gender, s.answer
                FROM theme_members m JOIN responses s ON s.id = m.response_id JOIN respondents r ON r.id = s.respondent_id
                WHERE m.theme_id = $1 ORDER BY m.distance, s.id LIMIT $2
                """,
                theme["id"], EXAMPLES,
            )
            entries.append({
                "kind": theme["kind"], "label": theme["label"], "description": theme["description"], "size": theme["size"],
                "share": round(theme["size"] / run["n_responses"], 3),
                "examples": [{"response_id": e["response_id"], "age": e["age"], "gender": e["gender"], "answer": _short(e["answer"], ANSWER_CHARS)} for e in examples],
            })
        question = json.loads(run["params"]).get("question")
        questions.append({
            "question_number": run["question_number"], "question": question, "responses": run["n_responses"],
            "silhouette": run["silhouette"], "themes": entries,
        })
    return {"study_id": study["id"], "brief": study["brief"], "questions": questions}


async def source_response(conn: asyncpg.Connection, response_id: int) -> dict | None:
    row = await conn.fetchrow(
        """
        SELECT s.id, s.question_number, s.question, s.answer, r.study_id, r.name, r.age, r.gender, r.city, r.occupation
        FROM responses s JOIN respondents r ON r.id = s.respondent_id WHERE s.id = $1
        """,
        response_id,
    )
    return dict(row) if row else None


async def agent_run(conn: asyncpg.Connection, run_id: int) -> dict | None:
    run = await conn.fetchrow(
        "SELECT id, question, model, status, answer, input_tokens, output_tokens, cost_usd, verified, problems, retried FROM agent_runs WHERE id = $1",
        run_id,
    )
    if run is None:
        return None
    steps = await conn.fetch(
        "SELECT step_number, kind, tool_name, arguments, result, is_error, input_tokens, output_tokens, latency_ms FROM agent_steps WHERE run_id = $1 ORDER BY step_number",
        run_id,
    )
    return {
        **{k: run[k] for k in ("id", "question", "model", "status", "answer", "input_tokens", "output_tokens", "verified", "retried")},
        "cost_usd": float(run["cost_usd"]) if run["cost_usd"] is not None else None,
        "problems": json.loads(run["problems"]),
        "steps": [
            {
                "number": s["step_number"], "kind": s["kind"], "tool": s["tool_name"],
                "arguments": json.loads(s["arguments"]) if s["arguments"] else None,
                "result": _short(s["result"], RESULT_CHARS), "is_error": s["is_error"],
                "tokens": s["input_tokens"] + s["output_tokens"], "latency_ms": round(s["latency_ms"]),
            }
            for s in steps
        ],
    }
