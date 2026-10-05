import asyncio
from types import SimpleNamespace as NS

from app import search as search_module
from app.embeddings import embed_responses
from app.search import SAMPLE_QUERIES, Neighbor, format_neighbors, nearest
from app.store import insert_simulation
from dbtools import in_rolled_back_transaction, simulation

DIMS = 1536
ANSWERS = {"a1": 0, "a2": 1, "b1": 2, "b2": 3, "c1": 4, "c2": 5}


def one_hot(position, also=None):
    """A vector with a 1 at `position`. `also` mixes in a second position at half strength."""
    vector = [0.0] * DIMS
    vector[position] = 1.0
    if also is not None:
        vector[also] = 0.5
    return vector


def embed_by_answer(stub_llm):
    async def embed_create(model, input):
        data = [NS(index=i, embedding=one_hot(ANSWERS[t])) for i, t in enumerate(input)]
        return NS(data=data, usage=NS(prompt_tokens=1))

    stub_llm.embed_create = embed_create


def test_neighbors_come_back_closest_first_and_limited_to_k(stub_llm):
    embed_by_answer(stub_llm)

    async def body(conn):
        study_id = await insert_simulation(conn, "a brief", 0, "m", simulation())
        await embed_responses(conn, "m", study_id)

        # closest to b2, then a bit to c1 (they share half a dimension), then everything else
        found = await nearest(conn, one_hot(3, also=4), "m", k=3)

        assert [n.answer for n in found][:2] == ["b2", "c1"]
        assert len(found) == 3
        assert found[0].distance < found[1].distance <= found[2].distance
        assert (found[0].study_id, found[0].question_number, found[0].age) == (study_id, 2, 40)

    in_rolled_back_transaction(body)


def test_vectors_from_other_models_are_never_compared(stub_llm):
    embed_by_answer(stub_llm)

    async def body(conn):
        study_id = await insert_simulation(conn, "a brief", 0, "m", simulation())
        await embed_responses(conn, "old-model", study_id)

        assert await nearest(conn, one_hot(0), "new-model", k=10) == []
        assert len(await nearest(conn, one_hot(0), "old-model", k=10)) == 6  # this study's six answers

    in_rolled_back_transaction(body)


def test_search_embeds_all_the_queries_in_one_request(stub_llm, monkeypatch):
    requests = []

    async def embed_create(model, input):
        requests.append(list(input))
        return NS(
            data=[NS(index=i, embedding=one_hot(i)) for i in range(len(input))],
            usage=NS(prompt_tokens=1),
        )

    stub_llm.embed_create = embed_create

    async def fake_connect():
        class Closed:
            async def close(self):
                pass

        return Closed()

    async def fake_nearest(conn, vector, model, k):
        return [Neighbor(0.1, 1, "brief", 1, f"hit for position {vector.index(1.0)}", None)]

    monkeypatch.setattr(search_module, "connect", fake_connect)
    monkeypatch.setattr(search_module, "nearest", fake_nearest)

    results = asyncio.run(search_module.search(["first", "second"], "m", 5))

    assert requests == [["first", "second"]]  # one API call, not one per query
    assert [n[0].answer for n in results.values()] == ["hit for position 0", "hit for position 1"]


def test_the_report_lists_studies_distances_and_shortened_answers():
    long_answer = "word " * 60
    results = {
        "too pricey": [
            Neighbor(0.25, 7, "A meal-kit startup wants to know why customers cancel", 4, long_answer, 34),
            Neighbor(0.5, 8, "A bank wants to learn about budgets", 2, "short answer", None),
        ]
    }

    report = format_neighbors(results)

    assert "7: A meal-kit startup wants to know why customers cancel" in report
    assert '1. dist 0.250  study 7 Q4, age 34: "word word' in report
    assert '2. dist 0.500  study 8 Q2: "short answer"' in report  # no age shown when unknown
    assert "..." in report and len(max(report.splitlines(), key=len)) < 200


def test_an_empty_result_explains_what_to_do():
    assert "make embed" in format_neighbors({"anything": []})


def test_there_are_five_distinct_sample_queries():
    assert len(SAMPLE_QUERIES) == 5 and len(set(SAMPLE_QUERIES)) == 5
