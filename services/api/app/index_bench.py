"""Compare exact search, an HNSW index and an IVFFlat index on speed and recall.

The test table is built from the real embeddings: every real vector gets several noisy copies, so the
vectors keep the real topic structure at a size where an index starts to matter. It is a scratch
table that is dropped at the end.
"""

import json
import time
from dataclasses import dataclass, field

import asyncpg

from app.db import connect
from app.embeddings import EMBEDDING_MODEL

K = 5
COPIES = 40
N_QUERIES = 50
NOISE = 0.10  # each dimension of a copy moves by up to +/- NOISE / 2. At 0.04 the copies were so tight that every index scored recall 1.0.


@dataclass
class Variant:
    label: str
    ms_per_query: float  # wall-clock average, including the round trip
    recall: float  # share of the exact top-K that this setup also returned
    plan: str  # the scan the planner chose for the first query
    explain_ms: float  # the server's own execution time for that query


@dataclass
class IndexInfo:
    label: str
    build_seconds: float
    size_mb: float


@dataclass
class BenchReport:
    rows: int
    real_vectors: int
    n_queries: int
    indexes: list[IndexInfo] = field(default_factory=list)
    variants: list[Variant] = field(default_factory=list)


def leaf_scan(plan: dict) -> str:
    """Name the deepest scan node in an EXPLAIN plan, such as 'Parallel Seq Scan' or 'Index Scan using x'."""
    node = plan
    found = node["Node Type"] if "Scan" in node["Node Type"] else ""
    while "Plans" in node:
        node = node["Plans"][0]
        if "Scan" in node["Node Type"]:
            found = node["Node Type"] + (f" using {node['Index Name']}" if "Index Name" in node else "")
    return found or plan["Node Type"]


def recall(exact: list[set[int]], approximate: list[set[int]]) -> float:
    hits = sum(len(e & a) for e, a in zip(exact, approximate, strict=True))
    return hits / sum(len(e) for e in exact)


async def _build_table(conn: asyncpg.Connection, model: str, copies: int) -> tuple[int, int]:
    await conn.execute("DROP TABLE IF EXISTS vector_bench")
    await conn.execute(
        "CREATE UNLOGGED TABLE vector_bench (id bigserial PRIMARY KEY, embedding vector(1536) NOT NULL)"
    )
    # The LATERAL subquery mentions s.id and the copy number, so Postgres draws new noise for every row.
    await conn.execute(
        """
        INSERT INTO vector_bench (embedding)
        SELECT s.embedding + n.noise
        FROM responses s
        CROSS JOIN generate_series(1, $2::int) AS copy
        CROSS JOIN LATERAL (
            SELECT array_agg(((random() - 0.5) * $3::float8)::real)::vector AS noise
            FROM generate_series(1, 1536) AS g
            WHERE g > -(s.id + copy)
        ) n
        WHERE s.embedding_model = $1
        """,
        model, copies, NOISE,
    )
    await conn.execute("ANALYZE vector_bench")
    rows = await conn.fetchval("SELECT count(*) FROM vector_bench")
    real = await conn.fetchval("SELECT count(*) FROM responses WHERE embedding_model = $1", model)
    return rows, real


_QUERY = "SELECT id FROM vector_bench ORDER BY embedding <=> $1::vector LIMIT $2"


async def _measure(conn: asyncpg.Connection, queries: list[str]) -> tuple[list[set[int]], float, str, float]:
    """Run every query. Returns the ids found, ms per query, the scan used and the server's own time."""
    start = time.perf_counter()
    found = [{row["id"] for row in await conn.fetch(_QUERY, q, K)} for q in queries]
    ms = (time.perf_counter() - start) * 1000 / len(queries)
    explained = json.loads(await conn.fetchval("EXPLAIN (ANALYZE, FORMAT JSON) " + _QUERY, queries[0], K))[0]
    return found, ms, leaf_scan(explained["Plan"]), explained["Execution Time"]


async def _index_size_mb(conn: asyncpg.Connection, index: str) -> float:
    return await conn.fetchval("SELECT pg_relation_size($1::regclass) / 1024.0 / 1024.0", index)


async def run_bench(
    conn: asyncpg.Connection,
    model: str = EMBEDDING_MODEL,
    copies: int = COPIES,
    n_queries: int = N_QUERIES,
) -> BenchReport:
    try:
        rows, real = await _build_table(conn, model, copies)
        queries = [
            row["v"]
            for row in await conn.fetch(
                "SELECT embedding::text AS v FROM responses WHERE embedding_model = $1 ORDER BY random() LIMIT $2",
                model, n_queries,
            )
        ]
        report = BenchReport(rows, real, len(queries))
        await conn.execute("SET maintenance_work_mem = '512MB'")  # lets the index graph build in memory

        exact, ms, plan, explain_ms = await _measure(conn, queries)
        report.variants.append(Variant("no index (exact)", ms, 1.0, plan, explain_ms))

        start = time.perf_counter()
        await conn.execute(
            "CREATE INDEX vector_bench_hnsw ON vector_bench USING hnsw (embedding vector_cosine_ops)"
        )
        report.indexes.append(
            IndexInfo("HNSW (m=16, ef_construction=64)", time.perf_counter() - start,
                      await _index_size_mb(conn, "vector_bench_hnsw"))
        )
        for ef in (10, 40, 100):
            await conn.execute(f"SET hnsw.ef_search = {ef}")
            found, ms, plan, explain_ms = await _measure(conn, queries)
            report.variants.append(Variant(f"HNSW ef_search={ef}", ms, recall(exact, found), plan, explain_ms))
        await conn.execute("DROP INDEX vector_bench_hnsw")

        lists = max(rows // 1000, 1)  # pgvector's rule of thumb for tables under a million rows
        start = time.perf_counter()
        await conn.execute(
            f"CREATE INDEX vector_bench_ivfflat ON vector_bench USING ivfflat (embedding vector_cosine_ops) WITH (lists = {lists})"
        )
        report.indexes.append(
            IndexInfo(f"IVFFlat (lists={lists})", time.perf_counter() - start,
                      await _index_size_mb(conn, "vector_bench_ivfflat"))
        )
        for probes in sorted({1, max(lists // 4, 1), max(lists // 2, 1)}):
            await conn.execute(f"SET ivfflat.probes = {probes}")
            found, ms, plan, explain_ms = await _measure(conn, queries)
            report.variants.append(
                Variant(f"IVFFlat probes={probes}", ms, recall(exact, found), plan, explain_ms)
            )
        return report
    finally:
        await conn.execute("DROP TABLE IF EXISTS vector_bench")


async def bench_index(model: str = EMBEDDING_MODEL, copies: int = COPIES) -> BenchReport:
    conn = await connect()
    try:
        return await run_bench(conn, model, copies)
    finally:
        await conn.close()


def format_bench(report: BenchReport) -> str:
    lines = [
        f"{report.rows:,} vectors x 1536 dims ({report.real_vectors} real embeddings with noisy copies), "
        f"{report.n_queries} queries, top {K}",
        "",
        f"{'index':<34}{'build':>9}{'size':>10}",
        *[f"{i.label:<34}{i.build_seconds:>8.1f}s{i.size_mb:>8.0f} MB" for i in report.indexes],
        "",
        f"{'setup':<24}{'ms/query':>10}{'recall@5':>10}{'server ms':>11}  plan",
        *[
            f"{v.label:<24}{v.ms_per_query:>10.2f}{v.recall:>10.2f}{v.explain_ms:>11.2f}  {v.plan}"
            for v in report.variants
        ],
    ]
    return "\n".join(lines)
