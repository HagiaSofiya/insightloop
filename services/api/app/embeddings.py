import hashlib
from dataclasses import dataclass

import asyncpg

from app.db import connect
from app.llm.client import embed

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dimensions, matching vector(1536) in the schema
BATCH_SIZE = 256  # texts per API request (the API accepts up to 2048)


@dataclass
class EmbedSummary:
    embedded: int = 0  # rows that received a vector
    cache_hits: int = 0  # distinct texts served from the cache
    api_texts: int = 0  # distinct texts sent to the API
    requests: int = 0
    tokens: int = 0
    cost_usd: float | None = 0.0


def to_vector_literal(vector: list[float]) -> str:
    """pgvector reads a vector from text like '[0.1,0.2,0.3]'. Postgres casts it with ::vector."""
    return "[" + ",".join(str(x) for x in vector) + "]"


def content_hash(text: str) -> str:
    """The cache key. SQL computes the same value with encode(sha256(convert_to(text, 'UTF8')), 'hex')."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def ensure_cached(
    conn: asyncpg.Connection, texts: list[str], model: str, batch_size: int = BATCH_SIZE
) -> EmbedSummary:
    """Make sure every text has a vector in embedding_cache, calling the API only for the missing ones.

    Each batch is saved as soon as it returns, so a run that fails part-way resumes where it stopped.
    """
    unique = list(dict.fromkeys(texts))  # drop duplicates, keep the order
    hashes = {text: content_hash(text) for text in unique}
    cached = {
        row["content_hash"]
        for row in await conn.fetch(
            "SELECT content_hash FROM embedding_cache WHERE model = $1 AND content_hash = ANY($2)",
            model, list(hashes.values()),
        )
    }
    missing = [text for text in unique if hashes[text] not in cached]
    summary = EmbedSummary(cache_hits=len(unique) - len(missing), api_texts=len(missing))

    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        result = await embed(batch, model)
        await conn.executemany(
            """
            INSERT INTO embedding_cache (content_hash, model, embedding) VALUES ($1, $2, $3::vector)
            ON CONFLICT DO NOTHING
            """,
            [
                (hashes[text], model, to_vector_literal(vector))
                for text, vector in zip(batch, result.vectors, strict=True)
            ],
        )
        summary.requests += 1
        summary.tokens += result.input_tokens
        summary.cost_usd = None if summary.cost_usd is None or result.cost_usd is None else summary.cost_usd + result.cost_usd
    return summary


async def embed_responses(
    conn: asyncpg.Connection, model: str = EMBEDDING_MODEL, study_id: int | None = None
) -> EmbedSummary:
    """Give every answer a vector from this model. Safe to run repeatedly."""
    rows = await conn.fetch(
        """
        SELECT s.id, s.answer
        FROM responses s JOIN respondents r ON r.id = s.respondent_id
        WHERE (s.embedding IS NULL OR s.embedding_model IS DISTINCT FROM $1)
          AND ($2::bigint IS NULL OR r.study_id = $2)
        ORDER BY s.id
        """,
        model, study_id,
    )
    if not rows:
        return EmbedSummary()

    summary = await ensure_cached(conn, [row["answer"] for row in rows], model)  # the answer text only
    await conn.execute(
        """
        UPDATE responses s SET embedding = e.embedding, embedding_model = $1
        FROM embedding_cache e
        WHERE s.id = ANY($2) AND e.model = $1
          AND e.content_hash = encode(sha256(convert_to(s.answer, 'UTF8')), 'hex')
        """,
        model, [row["id"] for row in rows],
    )
    summary.embedded = len(rows)
    return summary


async def embed_all(model: str = EMBEDDING_MODEL, study_id: int | None = None) -> EmbedSummary:
    conn = await connect()
    try:
        return await embed_responses(conn, model, study_id)
    finally:
        await conn.close()
