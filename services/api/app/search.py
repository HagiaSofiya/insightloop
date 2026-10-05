from dataclasses import dataclass

import asyncpg

from app.db import connect
from app.embeddings import EMBEDDING_MODEL, to_vector_literal
from app.llm.client import embed

SAMPLE_QUERIES = [
    "too expensive for what you get",
    "my knees and joints hurt",
    "tracking my spending and saving money",
    "the food went to waste",
    "I do not trust it with my personal information",
]


@dataclass
class Neighbor:
    distance: float  # cosine distance: 0 means the same direction, around 1 means unrelated
    study_id: int
    brief: str
    question_number: int
    answer: str
    age: int | None
    response_id: int = 0
    question: str = ""
    gender: str | None = None
    keyword_match: bool = False  # the response also matched the keyword query


async def nearest(
    conn: asyncpg.Connection,
    query_vector: list[float],
    model: str,
    k: int = 5,
    study_id: int | None = None,
    min_age: int | None = None,
    max_age: int | None = None,
    gender: str | None = None,
) -> list[Neighbor]:
    rows = await conn.fetch(
        """
        SELECT s.embedding <=> $1::vector AS distance, st.id AS study_id, st.brief,
               s.question_number, s.question, s.answer, s.id AS response_id, r.age, r.gender
        FROM responses s
        JOIN respondents r ON r.id = s.respondent_id
        JOIN studies st ON st.id = r.study_id
        WHERE s.embedding_model = $2          -- only compare vectors made by the same model
          AND ($4::bigint IS NULL OR st.id = $4)
          AND ($5::int IS NULL OR r.age >= $5)
          AND ($6::int IS NULL OR r.age <= $6)
          AND ($7::text IS NULL OR r.gender = $7)
        ORDER BY s.embedding <=> $1::vector
        LIMIT $3
        """,
        to_vector_literal(query_vector), model, k, study_id, min_age, max_age, gender,
    )
    return [
        Neighbor(
            row["distance"], row["study_id"], row["brief"], row["question_number"], row["answer"],
            row["age"], row["response_id"], row["question"], row["gender"],
        )
        for row in rows
    ]


async def search(
    queries: list[str], model: str = EMBEDDING_MODEL, k: int = 5
) -> dict[str, list[Neighbor]]:
    result = await embed(queries, model)  # all the queries in one request
    conn = await connect()
    try:
        return {
            query: await nearest(conn, vector, model, k)
            for query, vector in zip(queries, result.vectors, strict=True)
        }
    finally:
        await conn.close()


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_neighbors(results: dict[str, list[Neighbor]]) -> str:
    studies = {n.study_id: n.brief for neighbors in results.values() for n in neighbors}
    if not studies:
        return "No embedded responses found. Save a study with make respond, then run make embed."
    lines = ["studies:", *[f"  {sid}: {_shorten(brief, 70)}" for sid, brief in sorted(studies.items())]]
    for query, neighbors in results.items():
        lines += ["", f'query: "{query}"']
        for rank, n in enumerate(neighbors, 1):
            age = f", age {n.age}" if n.age is not None else ""
            lines.append(
                f"  {rank}. dist {n.distance:.3f}  study {n.study_id} Q{n.question_number}{age}: "
                f'"{_shorten(n.answer, 120)}"'
            )
    return "\n".join(lines)


def format_evidence(hits: list[Neighbor]) -> str:
    """Each response as a labeled block, for the model to read."""
    blocks = []
    for hit in hits:
        who = ", ".join(part for part in (hit.gender, f"age {hit.age}" if hit.age is not None else None) if part)
        blocks.append(f"[R{hit.response_id}] {who or 'unknown respondent'} | asked: {hit.question}\n{hit.answer}")
    return "\n\n".join(blocks)
