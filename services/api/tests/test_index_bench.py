import pytest

from app.embeddings import embed_responses
from app.index_bench import format_bench, leaf_scan, recall, run_bench
from app.store import insert_simulation
from dbtools import in_rolled_back_transaction, simulation


def test_recall_is_the_share_of_the_exact_neighbors_that_were_found():
    exact = [{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}]

    assert recall(exact, exact) == 1.0
    assert recall(exact, [{1, 2, 3, 4, 5}, {6, 7, 98, 99, 100}]) == pytest.approx(0.7)
    assert recall(exact, [{91, 92, 93, 94, 95}, {96, 97, 98, 99, 100}]) == 0.0


@pytest.mark.parametrize(
    "plan, expected",
    [
        ({"Node Type": "Limit", "Plans": [{"Node Type": "Index Scan", "Index Name": "idx"}]}, "Index Scan using idx"),
        (
            {"Node Type": "Limit", "Plans": [{"Node Type": "Gather Merge", "Plans": [
                {"Node Type": "Sort", "Plans": [{"Node Type": "Parallel Seq Scan"}]}]}]},
            "Parallel Seq Scan",
        ),
        ({"Node Type": "Result"}, "Result"),
    ],
    ids=["index", "parallel-seq", "no-scan"],
)
def test_the_scan_node_is_found_inside_the_plan(plan, expected):
    assert leaf_scan(plan) == expected


def test_the_benchmark_runs_end_to_end_on_a_tiny_table_and_cleans_up(stub_llm):
    from types import SimpleNamespace as NS

    async def embed_create(model, input):
        vectors = [[(i + 1.0) if d == i else 0.0 for d in range(1536)] for i in range(len(input))]
        return NS(data=[NS(index=i, embedding=v) for i, v in enumerate(vectors)], usage=NS(prompt_tokens=1))

    stub_llm.embed_create = embed_create

    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "tiny-model", simulation())
        await embed_responses(conn, "tiny-model", study_id)

        report = await run_bench(conn, "tiny-model", copies=3, n_queries=4)

        assert report.rows == 6 * 3 and report.real_vectors == 6
        labels = [v.label for v in report.variants]
        assert labels[0] == "no index (exact)" and any(l.startswith("HNSW") for l in labels)
        assert any(l.startswith("IVFFlat") for l in labels)
        assert report.variants[0].recall == 1.0  # exact search is its own reference
        assert all(0.0 <= v.recall <= 1.0 and v.ms_per_query > 0 for v in report.variants)
        assert [i.label.split()[0] for i in report.indexes] == ["HNSW", "IVFFlat"]
        assert await conn.fetchval("SELECT to_regclass('vector_bench')") is None  # the scratch table is gone
        text = format_bench(report)
        assert "recall@5" in text and "HNSW ef_search=40" in text

    in_rolled_back_transaction(body)
