"""Cluster one question's answers, name each cluster with the model and store the result."""

import json
from dataclasses import dataclass, field

import asyncpg
import numpy as np

from app.clustering import (
    OUTLIER, Clustering, cluster_hdbscan, cluster_kmeans, distances_to_centers, representatives,
)
from app.db import connect
from app.embeddings import EMBEDDING_MODEL
from app.llm.client import complete_structured
from app.prompts.registry import PromptRef, load
from app.schemas import GroupLabels

PROMPT_NAME = "theme_label"
PROMPT_VERSION = "v2"
LABEL_MODEL = "gpt-4o-mini"
QUOTES_FOR_LABEL = 5  # how many of each cluster's closest answers the model reads
QUOTE_CHARS = 300
MIN_RESPONSES = 8  # fewer answers than this is not enough to find themes


@dataclass
class Theme:
    kind: str  # "theme" or "outlier"
    label: str
    description: str
    response_ids: list[int]
    distances: list[float]  # each member's cosine distance to the theme's center


@dataclass
class ThemeRun:
    study_id: int
    question_number: int
    question: str
    method: str
    params: dict
    embedding_model: str
    n_responses: int
    silhouette: float | None
    themes: list[Theme] = field(default_factory=list)
    answers: dict[int, str] = field(default_factory=dict)  # response id -> text, for showing examples


async def _load(conn: asyncpg.Connection, study_id: int, question_number: int, model: str):
    rows = await conn.fetch(
        """
        SELECT s.id, s.answer, s.question, s.embedding::text AS vector, st.brief
        FROM responses s
        JOIN respondents r ON r.id = s.respondent_id
        JOIN studies st ON st.id = r.study_id
        WHERE r.study_id = $1 AND s.question_number = $2 AND s.embedding_model = $3
        ORDER BY s.id
        """,
        study_id, question_number, model,
    )
    if len(rows) < MIN_RESPONSES:
        raise ValueError(
            f"Study {study_id} question {question_number} has {len(rows)} embedded answers; "
            f"at least {MIN_RESPONSES} are needed to find themes."
        )
    X = np.array([json.loads(row["vector"]) for row in rows])
    return rows, X


async def label_themes(
    brief: str, question: str, groups: list[list[str]], model: str, response_ids: list[list[int]]
) -> dict[int, tuple[str, str]]:
    """Name every group in one call, so the model can see them side by side and keep the names distinct.

    Returns {group number (from 1): (label, description)}.
    """
    blocks = [
        f"Group {number}:\n" + "\n".join(f"- {' '.join(a.split())[:QUOTE_CHARS]}" for a in answers)
        for number, answers in enumerate(groups, 1)
    ]
    messages = [
        {"role": "system", "content": load(PROMPT_NAME, PROMPT_VERSION)},
        {"role": "user", "content": f"Study brief: {brief}\nQuestion asked: {question}\n\n" + "\n\n".join(blocks)},
    ]
    prompt = PromptRef(
        PROMPT_NAME, PROMPT_VERSION,
        {"brief": brief, "question": question, "response_ids": [i for group in response_ids for i in group]},
    )
    result: GroupLabels = await complete_structured(messages, model, GroupLabels, prompt)
    return {item.group: (item.label, item.description) for item in result.groups}


async def build_run(
    conn: asyncpg.Connection,
    study_id: int,
    question_number: int,
    method: str = "kmeans",
    k: int | None = None,
    min_cluster_size: int = 4,
    model: str = EMBEDDING_MODEL,
    label_model: str = LABEL_MODEL,
) -> ThemeRun:
    rows, X = await _load(conn, study_id, question_number, model)
    clustering: Clustering = (
        cluster_kmeans(X, k) if method == "kmeans" else cluster_hdbscan(X, min_cluster_size)
    )
    distances = distances_to_centers(X, clustering.labels)
    ids = [row["id"] for row in rows]
    brief, question = rows[0]["brief"], rows[0]["question"]

    clusters = range(clustering.n_clusters)
    closest = [representatives(distances, clustering.labels, c, QUOTES_FOR_LABEL) for c in clusters]
    named = await label_themes(
        brief, question, [[rows[i]["answer"] for i in group] for group in closest], label_model,
        [[ids[i] for i in group] for group in closest],
    ) if clustering.n_clusters else {}
    themes = []
    for cluster in clusters:
        members = [i for i, label in enumerate(clustering.labels) if label == cluster]
        label, description = named.get(cluster + 1, (f"Theme {cluster + 1}", "The model did not name this group."))
        themes.append(
            Theme("theme", label, description, [ids[i] for i in members], [float(distances[i]) for i in members])
        )
    outliers = [i for i, label in enumerate(clustering.labels) if label == OUTLIER]
    if outliers:
        themes.append(
            Theme("outlier", "Outliers", "Answers that fit none of the themes.",
                  [ids[i] for i in outliers], [float(distances[i]) for i in outliers])
        )
    params = {**clustering.params, "question": question}
    return ThemeRun(
        study_id, question_number, question, method, params, model, len(rows), clustering.silhouette,
        themes, {row["id"]: row["answer"] for row in rows},
    )


async def save_run(conn: asyncpg.Connection, run: ThemeRun) -> int:
    """Write the run, its themes and their members in one transaction. Returns the run id."""
    async with conn.transaction():
        run_id = await conn.fetchval(
            """
            INSERT INTO theme_runs (study_id, question_number, method, params, embedding_model, n_responses, silhouette)
            VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7) RETURNING id
            """,
            run.study_id, run.question_number, run.method, json.dumps(run.params),
            run.embedding_model, run.n_responses, run.silhouette,
        )
        for theme in run.themes:
            theme_id = await conn.fetchval(
                "INSERT INTO themes (run_id, kind, label, description, size) VALUES ($1, $2, $3, $4, $5) RETURNING id",
                run_id, theme.kind, theme.label, theme.description, len(theme.response_ids),
            )
            await conn.executemany(
                "INSERT INTO theme_members (theme_id, response_id, distance) VALUES ($1, $2, $3)",
                [(theme_id, rid, d) for rid, d in zip(theme.response_ids, theme.distances, strict=True)],
            )
    return run_id


async def themes_for_question(
    study_id: int, question_number: int, method: str = "kmeans", k: int | None = None, save: bool = True
) -> tuple[ThemeRun, int | None]:
    conn = await connect()
    try:
        run = await build_run(conn, study_id, question_number, method, k)
        return run, (await save_run(conn, run) if save else None)
    finally:
        await conn.close()


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_run(run: ThemeRun, examples: int = 2) -> str:
    quality = f"silhouette {run.silhouette:.2f}" if run.silhouette is not None else "silhouette n/a"
    lines = [
        f'study {run.study_id} Q{run.question_number}: "{_short(run.question, 90)}"',
        f"{run.method}, {run.n_responses} responses, {quality}",
    ]
    for number, theme in enumerate((t for t in run.themes if t.kind == "theme"), 1):
        share = len(theme.response_ids) / run.n_responses
        lines.append(f"  {number}. {theme.label} ({len(theme.response_ids)}, {share:.0%}): {theme.description}")
        closest = sorted(zip(theme.distances, theme.response_ids))[:examples]
        lines += [f'       [R{rid}] "{_short(run.answers[rid], 130)}"' for _, rid in closest]
    for theme in (t for t in run.themes if t.kind == "outlier"):
        lines.append(f"  Outliers ({len(theme.response_ids)}): fit none of the themes")
        closest = sorted(zip(theme.distances, theme.response_ids))[:1]
        lines += [f'       [R{rid}] "{_short(run.answers[rid], 130)}"' for _, rid in closest]
    return "\n".join(lines)
