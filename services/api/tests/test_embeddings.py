import asyncio
from types import SimpleNamespace as NS

import httpx2 as httpx
import openai
import pytest

from app.embeddings import content_hash, embed_responses, ensure_cached, to_vector_literal
from app.llm.client import embed
from app.llm.pricing import estimate_embedding_cost
from app.store import insert_simulation
from dbtools import in_rolled_back_transaction, simulation

DIMS = 1536  # the size of the vector column


def one_hot(position):
    """A vector with a single 1. Two different ones are as far apart as cosine distance allows."""
    vector = [0.0] * DIMS
    vector[position] = 1.0
    return vector


def script_embeddings(stub_llm, make_vector, tokens=10, reverse=False):
    """Make the stub answer embedding requests, and record the texts it was asked about."""
    stub_llm.embedded_texts = []

    async def embed_create(model, input):
        stub_llm.embedded_texts.append(list(input))
        data = [NS(index=i, embedding=make_vector(text)) for i, text in enumerate(input)]
        return NS(data=list(reversed(data)) if reverse else data, usage=NS(prompt_tokens=tokens))

    stub_llm.embed_create = embed_create


def test_vectors_come_back_in_input_order_even_if_the_response_is_shuffled(stub_llm):
    script_embeddings(stub_llm, lambda text: [float(len(text))], reverse=True)

    result = asyncio.run(embed(["a", "bb", "ccc"], "text-embedding-3-small"))

    assert result.vectors == [[1.0], [2.0], [3.0]]


def test_cost_is_tokens_times_the_model_price(stub_llm):
    script_embeddings(stub_llm, lambda text: [0.0], tokens=500_000)

    small = asyncio.run(embed(["x"], "text-embedding-3-small"))
    unknown = asyncio.run(embed(["x"], "some-new-model"))

    assert small.input_tokens == 500_000
    assert small.cost_usd == pytest.approx(0.01)  # 0.5M tokens at $0.02 per 1M
    assert unknown.cost_usd is None
    assert estimate_embedding_cost("text-embedding-3-large", 1_000_000) == pytest.approx(0.13)


def test_a_rate_limited_embedding_call_is_retried(stub_llm, sleeps):
    calls = {"n": 0}

    async def embed_create(model, input):
        calls["n"] += 1
        if calls["n"] == 1:
            response = httpx.Response(429, request=httpx.Request("POST", "http://test"))
            raise openai.RateLimitError("slow down", response=response, body={"type": "requests"})
        return NS(data=[NS(index=0, embedding=[1.0])], usage=NS(prompt_tokens=1))

    stub_llm.embed_create = embed_create

    result = asyncio.run(embed(["x"], "m"))

    assert result.vectors == [[1.0]] and calls["n"] == 2 and len(sleeps) == 1


def test_vectors_are_written_as_pgvector_text():
    assert to_vector_literal([0.5, -1.0, 2e-05]) == "[0.5,-1.0,2e-05]"


def test_every_answer_is_embedded_once_using_the_answer_text_only(stub_llm):
    answers = {"a1": 0, "a2": 1, "b1": 2, "b2": 3, "c1": 4, "c2": 5}
    script_embeddings(stub_llm, lambda text: one_hot(answers[text]))

    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())

        first = await embed_responses(conn, "text-embedding-3-small", study_id)

        assert first.embedded == 6 and first.tokens == 10
        assert stub_llm.embedded_texts == [["a1", "a2", "b1", "b2", "c1", "c2"]]  # no "Q:" prefixes
        stored = await conn.fetch(
            """
            SELECT s.answer, s.embedding_model, vector_dims(s.embedding) AS dims
            FROM responses s JOIN respondents r ON r.id = s.respondent_id
            WHERE r.study_id = $1 ORDER BY s.id
            """,
            study_id,
        )
        assert {(row["embedding_model"], row["dims"]) for row in stored} == {("text-embedding-3-small", DIMS)}

        second = await embed_responses(conn, "text-embedding-3-small", study_id)

        assert second.embedded == 0 and len(stub_llm.embedded_texts) == 1  # nothing to do, so no API call

    in_rolled_back_transaction(body)


def test_a_new_model_re_embeds_everything_because_vectors_from_different_models_do_not_mix(stub_llm):
    script_embeddings(stub_llm, lambda text: one_hot(0))

    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())
        await embed_responses(conn, "model-one", study_id)

        again = await embed_responses(conn, "model-two", study_id)

        assert again.embedded == 6
        models = await conn.fetch(
            """
            SELECT DISTINCT s.embedding_model FROM responses s
            JOIN respondents r ON r.id = s.respondent_id WHERE r.study_id = $1
            """,
            study_id,
        )
        assert [row["embedding_model"] for row in models] == ["model-two"]

    in_rolled_back_transaction(body)


def test_the_nearest_neighbor_query_finds_the_matching_answer(stub_llm):
    answers = {"a1": 0, "a2": 1, "b1": 2, "b2": 3, "c1": 4, "c2": 5}
    script_embeddings(stub_llm, lambda text: one_hot(answers[text]))

    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())
        await embed_responses(conn, "m", study_id)

        nearest = await conn.fetch(
            """
            SELECT s.answer, s.embedding <=> $1::vector AS distance
            FROM responses s JOIN respondents r ON r.id = s.respondent_id
            WHERE r.study_id = $2
            ORDER BY s.embedding <=> $1::vector LIMIT 2
            """,
            to_vector_literal(one_hot(3)), study_id,
        )

        assert nearest[0]["answer"] == "b2" and nearest[0]["distance"] == pytest.approx(0)
        assert nearest[1]["distance"] == pytest.approx(1)  # unrelated one-hot vectors are at cosine distance 1

    in_rolled_back_transaction(body)


# ---- batching and the cache -------------------------------------------------------------------


def test_the_hash_matches_the_one_postgres_computes():
    async def body(conn):
        for text in ["plain", "naïve café – ünïcode", "line one\nline two", "emoji 🙂"]:
            in_sql = await conn.fetchval("SELECT encode(sha256(convert_to($1, 'UTF8')), 'hex')", text)
            assert in_sql == content_hash(text)

    in_rolled_back_transaction(body)


def test_identical_answers_are_embedded_once_and_both_rows_get_the_vector(stub_llm):
    script_embeddings(stub_llm, lambda text: one_hot(0))

    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())
        await conn.execute("UPDATE responses SET answer = 'same words' WHERE answer IN ('a1', 'b1')")

        summary = await embed_responses(conn, "m", study_id)

        assert summary.embedded == 6 and summary.api_texts == 5  # six rows, five distinct texts
        assert stub_llm.embedded_texts == [["same words", "a2", "b2", "c1", "c2"]]
        assert await conn.fetchval(
            "SELECT count(*) FROM responses WHERE answer = 'same words' AND embedding IS NOT NULL"
        ) == 2

    in_rolled_back_transaction(body)


def test_clearing_the_vectors_costs_nothing_because_the_cache_remembers(stub_llm):
    script_embeddings(stub_llm, lambda text: one_hot(0))

    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())
        await embed_responses(conn, "m", study_id)
        in_this_study = "respondent_id IN (SELECT id FROM respondents WHERE study_id = $1)"
        await conn.execute(
            f"UPDATE responses SET embedding = NULL, embedding_model = NULL WHERE {in_this_study}", study_id
        )

        again = await embed_responses(conn, "m", study_id)

        assert again.embedded == 6 and again.cache_hits == 6 and again.api_texts == 0
        assert again.requests == 0 and len(stub_llm.embedded_texts) == 1  # still just the first request
        assert await conn.fetchval(
            f"SELECT count(*) FROM responses WHERE embedding IS NULL AND {in_this_study}", study_id
        ) == 0

    in_rolled_back_transaction(body)


def test_the_cache_is_per_model(stub_llm):
    script_embeddings(stub_llm, lambda text: one_hot(0))

    async def body(conn):
        await ensure_cached(conn, ["hello"], "model-one")

        other = await ensure_cached(conn, ["hello"], "model-two")
        same = await ensure_cached(conn, ["hello"], "model-one")

        assert (other.cache_hits, other.api_texts) == (0, 1)
        assert (same.cache_hits, same.api_texts) == (1, 0)

    in_rolled_back_transaction(body)


def test_texts_are_sent_in_batches(stub_llm):
    script_embeddings(stub_llm, lambda text: one_hot(0), tokens=7)

    async def body(conn):
        texts = [f"text {n}" for n in range(5)]

        summary = await ensure_cached(conn, texts, "m", batch_size=2)

        assert [len(batch) for batch in stub_llm.embedded_texts] == [2, 2, 1]
        assert sum(stub_llm.embedded_texts, []) == texts  # nothing lost, nothing repeated, order kept
        assert (summary.requests, summary.tokens) == (3, 21)

    in_rolled_back_transaction(body)


def test_a_failed_batch_keeps_the_earlier_batches_and_the_rerun_sends_only_the_rest(stub_llm):
    state = {"calls": 0}
    sent = []

    async def embed_create(model, input):
        state["calls"] += 1
        sent.append(list(input))
        if state["calls"] == 2:
            raise ValueError("the network died")
        return NS(
            data=[NS(index=i, embedding=one_hot(0)) for i in range(len(input))],
            usage=NS(prompt_tokens=1),
        )

    stub_llm.embed_create = embed_create

    async def body(conn):
        texts = [f"text {n}" for n in range(5)]
        with pytest.raises(ValueError):
            await ensure_cached(conn, texts, "m", batch_size=2)
        assert await conn.fetchval("SELECT count(*) FROM embedding_cache WHERE model = 'm'") == 2

        sent.clear()
        rerun = await ensure_cached(conn, texts, "m", batch_size=2)

        assert rerun.cache_hits == 2 and rerun.api_texts == 3
        assert sum(sent, []) == ["text 2", "text 3", "text 4"]  # only what was missing

    in_rolled_back_transaction(body)
