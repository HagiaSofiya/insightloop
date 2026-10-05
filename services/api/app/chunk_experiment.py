"""Does splitting long answers into chunks find specific ideas better than one vector per answer?

Each strategy returns its 5 best distinct responses for a concept query. A response counts as a hit if
its full text mentions the concept (a keyword check). The same check is used for every strategy, so
longer or shorter retrieved text does not favor one of them.
"""

import re
from dataclasses import dataclass, field
from statistics import mean

import asyncpg

from app.chunking import STRATEGIES as CHUNKERS
from app.db import connect
from app.embeddings import EMBEDDING_MODEL, EmbedSummary, embed_responses, ensure_cached, to_vector_literal
from app.llm.client import embed

K = 5
STRATEGY_NAMES = ["whole", *CHUNKERS]  # "whole" is the existing one-vector-per-answer embedding

# (the query to search for, a pattern that shows a response really mentions it)
CONCEPTS = [
    ("worried about the security of my personal information", r"secur|privacy|trust|safe|personal (info|data)|hack|fraud"),
    ("the food spoiled or got thrown away", r"waste|spoil|throw|threw|rot"),
    ("pain or injury from running", r"injur|pain|knee|joint|hurt|sore|ache"),
    ("it costs too much", r"cost|expens|price|pricey|afford|\$|money|budget"),
    ("the app was slow or buggy", r"crash|bug|glitch|freez|slow|laggy|error"),
    ("the delivery was late or missing", r"deliver|late|arriv|shipping"),
]


@dataclass
class Hit:
    distance: float
    text: str  # what was matched: the whole answer or one chunk
    answer: str  # the whole answer it belongs to


@dataclass
class ConceptResult:
    query: str
    mention_rate: dict[str, float]  # strategy -> share of the 5 hits whose answer mentions the concept
    best: dict[str, Hit]  # strategy -> its top hit


@dataclass
class ExperimentReport:
    chunk_counts: dict[str, int] = field(default_factory=dict)
    embedding: dict[str, EmbedSummary] = field(default_factory=dict)
    concepts: list[ConceptResult] = field(default_factory=list)
    avg_chars: dict[str, float] = field(default_factory=dict)


async def build_chunks(conn: asyncpg.Connection, strategy: str, study_id: int | None = None) -> int:
    """Create the chunks for responses that have none for this strategy yet. Returns how many were added."""
    rows = await conn.fetch(
        """
        SELECT s.id, s.answer FROM responses s JOIN respondents r ON r.id = s.respondent_id
        WHERE NOT EXISTS (SELECT 1 FROM response_chunks c WHERE c.response_id = s.id AND c.strategy = $1)
          AND ($2::bigint IS NULL OR r.study_id = $2)
        """,
        strategy, study_id,
    )
    records = [
        (row["id"], strategy, index, text)
        for row in rows
        for index, text in enumerate(CHUNKERS[strategy](row["answer"]))
    ]
    await conn.executemany(
        """
        INSERT INTO response_chunks (response_id, strategy, chunk_index, text) VALUES ($1, $2, $3, $4)
        ON CONFLICT DO NOTHING
        """,
        records,
    )
    return len(records)


async def embed_chunks(
    conn: asyncpg.Connection, strategy: str, model: str, study_id: int | None = None
) -> EmbedSummary:
    rows = await conn.fetch(
        """
        SELECT c.id, c.text FROM response_chunks c
        JOIN responses s ON s.id = c.response_id JOIN respondents r ON r.id = s.respondent_id
        WHERE c.strategy = $1 AND (c.embedding IS NULL OR c.embedding_model IS DISTINCT FROM $2)
          AND ($3::bigint IS NULL OR r.study_id = $3)
        ORDER BY c.id
        """,
        strategy, model, study_id,
    )
    if not rows:
        return EmbedSummary()
    summary = await ensure_cached(conn, [row["text"] for row in rows], model)
    await conn.execute(
        """
        UPDATE response_chunks c SET embedding = e.embedding, embedding_model = $2
        FROM embedding_cache e
        WHERE c.id = ANY($1) AND e.model = $2
          AND e.content_hash = encode(sha256(convert_to(c.text, 'UTF8')), 'hex')
        """,
        [row["id"] for row in rows], model,
    )
    summary.embedded = len(rows)
    return summary


async def top_responses(
    conn: asyncpg.Connection, strategy: str, query_vector: list[float], model: str, k: int = K
) -> list[Hit]:
    """The k closest distinct responses. A chunked response is represented by its closest chunk."""
    vector = to_vector_literal(query_vector)
    if strategy == "whole":
        rows = await conn.fetch(
            """
            SELECT s.embedding <=> $1::vector AS distance, s.answer AS text, s.answer
            FROM responses s WHERE s.embedding_model = $2
            ORDER BY s.embedding <=> $1::vector LIMIT $3
            """,
            vector, model, k,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT distance, text, answer FROM (
                SELECT DISTINCT ON (c.response_id)
                       c.embedding <=> $1::vector AS distance, c.text, s.answer
                FROM response_chunks c JOIN responses s ON s.id = c.response_id
                WHERE c.strategy = $2 AND c.embedding_model = $3
                ORDER BY c.response_id, c.embedding <=> $1::vector
            ) best
            ORDER BY distance LIMIT $4
            """,
            vector, strategy, model, k,
        )
    return [Hit(row["distance"], row["text"], row["answer"]) for row in rows]


def mention_rate(hits: list[Hit], pattern: str) -> float:
    return sum(bool(re.search(pattern, hit.answer, re.IGNORECASE)) for hit in hits) / len(hits) if hits else 0.0


async def run_experiment(
    conn: asyncpg.Connection, model: str = EMBEDDING_MODEL, study_id: int | None = None
) -> ExperimentReport:
    report = ExperimentReport()
    await embed_responses(conn, model, study_id)  # the whole-answer vectors
    for strategy in CHUNKERS:
        await build_chunks(conn, strategy, study_id)
        report.embedding[strategy] = await embed_chunks(conn, strategy, model, study_id)
        report.chunk_counts[strategy] = await conn.fetchval(
            """
            SELECT count(*) FROM response_chunks c JOIN responses s ON s.id = c.response_id
            JOIN respondents r ON r.id = s.respondent_id
            WHERE c.strategy = $1 AND ($2::bigint IS NULL OR r.study_id = $2)
            """,
            strategy, study_id,
        )

    queries = [query for query, _ in CONCEPTS]
    vectors = (await embed(queries, model)).vectors
    lengths: dict[str, list[int]] = {name: [] for name in STRATEGY_NAMES}
    for (query, pattern), vector in zip(CONCEPTS, vectors, strict=True):
        hits = {name: await top_responses(conn, name, vector, model) for name in STRATEGY_NAMES}
        for name, found in hits.items():
            lengths[name] += [len(hit.text) for hit in found]
        report.concepts.append(
            ConceptResult(
                query,
                {name: mention_rate(found, pattern) for name, found in hits.items()},
                {name: found[0] for name, found in hits.items() if found},
            )
        )
    report.avg_chars = {name: mean(sizes) for name, sizes in lengths.items() if sizes}
    return report


async def chunk_compare(model: str = EMBEDDING_MODEL) -> ExperimentReport:
    conn = await connect()
    try:
        return await run_experiment(conn, model)
    finally:
        await conn.close()


def _short(text: str, limit: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def format_experiment(report: ExperimentReport) -> str:
    names = STRATEGY_NAMES
    lines = ["chunks created:  " + "   ".join(f"{s} {n:,}" for s, n in report.chunk_counts.items())]
    for strategy, summary in report.embedding.items():
        cost = f"${summary.cost_usd:.4f}" if summary.cost_usd is not None else "unknown cost"
        lines.append(
            f"embedding {strategy}: {summary.api_texts} texts sent in {summary.requests} requests, "
            f"{summary.cache_hits} from the cache, {summary.tokens:,} tokens, {cost}"
        )
    lines += [
        "",
        f"share of the {K} best responses that really mention the concept (keyword check)",
        f"{'concept':<56}" + "".join(f"{n:>10}" for n in names),
    ]
    for result in report.concepts:
        lines.append(f"{_short(result.query, 54):<56}" + "".join(f"{result.mention_rate[n]:>10.2f}" for n in names))
    lines.append(
        f"{'mean':<56}"
        + "".join(f"{mean(r.mention_rate[n] for r in report.concepts):>10.2f}" for n in names)
    )
    lines.append(f"{'text matched per hit (avg characters)':<56}" + "".join(f"{report.avg_chars.get(n, 0):>10.0f}" for n in names))

    gains = [
        (max(r.mention_rate[n] for n in names[1:]) - r.mention_rate["whole"], r) for r in report.concepts
    ]
    gain, example = max(gains, key=lambda pair: pair[0])
    lines += ["", f'example, the concept where chunking helped most: "{example.query}"']
    for name in names:
        hit = example.best.get(name)
        if hit:
            lines.append(f"  {name:<9} dist {hit.distance:.3f}  {_short(hit.text)}")
    return "\n".join(lines)
