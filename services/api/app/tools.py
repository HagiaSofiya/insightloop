"""The tools the agent can call. Each validates its arguments, queries the database and returns plain data.

Anything the model can fix by changing its request raises ToolError, and run_tool turns that into an
error message the model reads and acts on. A real bug (a crash in our own code) is not caught here.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

import asyncpg
from pydantic import BaseModel, Field, ValidationError

from app.llm.client import embed
from app.retrieval import hybrid_candidates

MAX_RESULT_CHARS = 6000  # a bigger result would crowd out the conversation, so the model is told to narrow down
ANSWER_CHARS = 300  # per answer in search results and theme examples
QUOTE_CHARS = 600  # per answer from get_quotes
MAX_DISTANCE = 0.50  # the same threshold as /ask: closer than this counts as a strong match


class ToolError(Exception):
    """The request was wrong in a way the model can fix. The message is sent back to it."""


@dataclass
class ToolContext:
    conn: asyncpg.Connection
    embedding_model: str


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


async def _check_study(conn: asyncpg.Connection, study_id: int) -> None:
    ids = [row["id"] for row in await conn.fetch("SELECT id FROM studies ORDER BY id")]
    if study_id not in ids:
        raise ToolError(f"There is no study with id {study_id}. Study ids: {ids}.")


# ---- search_responses -------------------------------------------------------------------------


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, description="What to look for, in plain words, for example 'price too high'.")
    study_id: int | None = Field(None, description="Only this study. Leave empty to search every study.")
    min_age: int | None = Field(None, ge=0, le=120, description="Only respondents this age or older.")
    max_age: int | None = Field(None, ge=0, le=120, description="Only respondents this age or younger.")
    gender: Literal["woman", "man", "non-binary"] | None = Field(None, description="Only respondents of this gender.")
    limit: int = Field(5, ge=1, le=10, description="How many responses to return, at most 10.")


async def search_responses(ctx: ToolContext, args: SearchArgs) -> dict:
    if args.min_age is not None and args.max_age is not None and args.min_age > args.max_age:
        raise ToolError(f"min_age ({args.min_age}) is greater than max_age ({args.max_age}), so nobody can match.")
    if args.study_id is not None:
        await _check_study(ctx.conn, args.study_id)
    vector = (await embed([args.query], ctx.embedding_model)).vectors[0]
    found = await hybrid_candidates(
        ctx.conn, args.query, vector, ctx.embedding_model, args.study_id,
        min_age=args.min_age, max_age=args.max_age, gender=args.gender,
    )
    results = [
        {
            "response_id": hit.response_id,
            "study_id": hit.study_id,
            "question_number": hit.question_number,
            "age": hit.age,
            "gender": hit.gender,
            "match": "strong" if hit.distance <= MAX_DISTANCE or hit.keyword_match else "weak",
            "answer": _short(hit.answer, ANSWER_CHARS),
        }
        for hit in found.hits[: args.limit]
    ]
    note = "No responses matched these filters." if not results else None
    return {"count": len(results), "results": results, **({"note": note} if note else {})}


# ---- get_theme and count_by_theme -------------------------------------------------------------


async def _latest_run(conn: asyncpg.Connection, study_id: int, question_number: int) -> asyncpg.Record:
    await _check_study(conn, study_id)
    run = await conn.fetchrow(
        "SELECT id, n_responses FROM theme_runs WHERE study_id = $1 AND question_number = $2 ORDER BY id DESC LIMIT 1",
        study_id, question_number,
    )
    if run is None:
        available = await conn.fetch(
            "SELECT study_id, array_agg(DISTINCT question_number ORDER BY question_number) AS questions "
            "FROM theme_runs GROUP BY study_id ORDER BY study_id"
        )
        listing = "; ".join(f"study {r['study_id']}: questions {list(r['questions'])}" for r in available) or "none"
        raise ToolError(f"No themes exist for study {study_id} question {question_number}. Themes exist for {listing}.")
    return run


class GetThemeArgs(BaseModel):
    study_id: int
    question_number: int = Field(ge=1, description="The question's number within the study, starting at 1.")
    theme: str | None = Field(None, description="Part of a theme's label. Leave empty to list every theme of the question.")


async def get_theme(ctx: ToolContext, args: GetThemeArgs) -> dict:
    run = await _latest_run(ctx.conn, args.study_id, args.question_number)
    themes = await ctx.conn.fetch(
        "SELECT id, kind, label, description, size FROM themes WHERE run_id = $1 ORDER BY (kind = 'outlier'), size DESC, id",
        run["id"],
    )
    listing = [
        {"label": t["label"], "description": t["description"], "size": t["size"], "share": round(t["size"] / run["n_responses"], 2)}
        for t in themes
    ]
    if args.theme is None:
        return {"study_id": args.study_id, "question_number": args.question_number, "responses": run["n_responses"], "themes": listing}

    wanted = args.theme.lower()
    matches = [t for t in themes if wanted in t["label"].lower()]
    if not matches:
        raise ToolError(f"No theme label contains '{args.theme}'. The labels are: {[t['label'] for t in themes]}.")
    theme = matches[0]
    members = await ctx.conn.fetch(
        """
        SELECT s.id AS response_id, r.age, r.gender, s.answer
        FROM theme_members m JOIN responses s ON s.id = m.response_id JOIN respondents r ON r.id = s.respondent_id
        WHERE m.theme_id = $1 ORDER BY m.distance, s.id LIMIT 5
        """,
        theme["id"],
    )
    return {
        "theme": next(item for item in listing if item["label"] == theme["label"]),
        "examples": [
            {"response_id": m["response_id"], "age": m["age"], "gender": m["gender"], "answer": _short(m["answer"], ANSWER_CHARS)}
            for m in members
        ],
    }


# The group expressions are fixed strings chosen by the enum below, never text from the model.
GROUPS = {
    "age_band": "CASE WHEN r.age IS NULL THEN 'unknown' WHEN r.age < 30 THEN '18-29' WHEN r.age < 45 THEN '30-44' "
    "WHEN r.age < 60 THEN '45-59' ELSE '60+' END",
    "gender": "coalesce(r.gender, 'unknown')",
    "none": "'all'",
}


class CountArgs(BaseModel):
    study_id: int
    question_number: int = Field(ge=1, description="The question's number within the study, starting at 1.")
    group_by: Literal["age_band", "gender", "none"] = Field(
        "age_band", description="Split each theme's count by age band (18-29, 30-44, 45-59, 60+), by gender, or not at all."
    )


async def count_by_theme(ctx: ToolContext, args: CountArgs) -> dict:
    run = await _latest_run(ctx.conn, args.study_id, args.question_number)
    rows = await ctx.conn.fetch(
        f"""
        SELECT t.id, t.label, t.size, {GROUPS[args.group_by]} AS grp, count(*) AS n
        FROM themes t
        JOIN theme_members m ON m.theme_id = t.id
        JOIN responses s ON s.id = m.response_id
        JOIN respondents r ON r.id = s.respondent_id
        WHERE t.run_id = $1
        GROUP BY t.id, t.label, t.size, grp
        ORDER BY t.size DESC, t.id, grp
        """,
        run["id"],
    )
    themes: dict[int, dict] = {}
    totals: dict[str, int] = {}
    for row in rows:
        entry = themes.setdefault(row["id"], {"theme": row["label"], "size": row["size"], "counts": {}})
        entry["counts"][row["grp"]] = row["n"]
        totals[row["grp"]] = totals.get(row["grp"], 0) + row["n"]
    return {
        "study_id": args.study_id,
        "question_number": args.question_number,
        "group_by": args.group_by,
        "responses": run["n_responses"],
        "themes": list(themes.values()),
        "totals": dict(sorted(totals.items())),
    }


# ---- get_quotes -------------------------------------------------------------------------------


class QuotesArgs(BaseModel):
    response_ids: list[int] = Field(min_length=1, max_length=5, description="Between 1 and 5 response ids from other tool results.")


async def get_quotes(ctx: ToolContext, args: QuotesArgs) -> dict:
    rows = await ctx.conn.fetch(
        """
        SELECT s.id, r.study_id, s.question_number, s.question, s.answer, r.age, r.gender
        FROM responses s JOIN respondents r ON r.id = s.respondent_id WHERE s.id = ANY($1)
        """,
        args.response_ids,
    )
    by_id = {row["id"]: row for row in rows}
    missing = [i for i in args.response_ids if i not in by_id]
    if missing:
        raise ToolError(f"No responses with ids {missing}. Use ids returned by search_responses or get_theme.")
    return {
        "quotes": [
            {
                "response_id": i, "study_id": by_id[i]["study_id"], "question_number": by_id[i]["question_number"],
                "question": by_id[i]["question"], "age": by_id[i]["age"], "gender": by_id[i]["gender"],
                "answer": _short(by_id[i]["answer"], QUOTE_CHARS),
            }
            for i in args.response_ids
        ]
    }


# ---- the registry -----------------------------------------------------------------------------


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    handler: Callable[[ToolContext, Any], Awaitable[dict]]


TOOLS: dict[str, Tool] = {
    tool.name: tool
    for tool in [
        Tool(
            "search_responses",
            "Find the respondent answers most relevant to a topic, optionally limited to a study, an age range or a gender. "
            "Each result says whether it is a strong or weak match.",
            SearchArgs, search_responses,
        ),
        Tool(
            "get_theme",
            "List the themes found in one question's answers, or show one theme with example answers.",
            GetThemeArgs, get_theme,
        ),
        Tool(
            "count_by_theme",
            "Count how many respondents fall in each theme of a question, split by age band or gender. "
            "Use this for any 'how many' or 'how common' claim.",
            CountArgs, count_by_theme,
        ),
        Tool(
            "get_quotes",
            "Get the full text of up to 5 specific responses by id, so you can quote them exactly.",
            QuotesArgs, get_quotes,
        ),
    ]
}


def tool_specs() -> list[dict]:
    """The tools in the format the chat API expects."""
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.args.model_json_schema()}}
        for t in TOOLS.values()
    ]


def _describe(error: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(part) for part in e['loc']) or 'arguments'}: {e['msg']}" for e in error.errors())


async def run_tool(name: str, arguments: str, ctx: ToolContext) -> tuple[str, bool]:
    """Run one tool call. Returns (result as JSON text, whether it was an error the model should fix)."""
    tool = TOOLS.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool '{name}'. The available tools are: {list(TOOLS)}."}), True
    try:
        try:
            raw = json.loads(arguments or "{}")
        except json.JSONDecodeError as error:
            raise ToolError(f"The arguments are not valid JSON ({error.msg}).")
        if not isinstance(raw, dict):
            raise ToolError("The arguments must be a JSON object.")
        result = json.dumps(await tool.handler(ctx, tool.args(**raw)), ensure_ascii=False)
        if len(result) > MAX_RESULT_CHARS:
            raise ToolError(f"The result is too large ({len(result):,} characters). Ask for less, for example a smaller limit or a narrower filter.")
        return result, False
    except ValidationError as error:
        return json.dumps({"error": f"Invalid arguments: {_describe(error)}"}), True
    except ToolError as error:
        return json.dumps({"error": str(error)}), True
