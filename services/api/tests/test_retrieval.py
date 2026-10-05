"""Tests for keyword search, fusion, hybrid retrieval and rerank.

The test answers use made-up words ("quizzlewump", "glorptastic"), so the results never depend on
what is in the real responses table.
"""

import asyncio
import math
from types import SimpleNamespace as NS

import pytest

from app import retrieval
from app.ask import ask
from app.rerank import rerank
from app.retrieval import hybrid_candidates, keyword_search, keyword_terms, reciprocal_rank_fusion
from app.retrieval_eval import EvalReport, QuestionResult, format_eval, precision_at_k
from app.schemas import Answer, Citation, Relevance, Reranking
from app.search import Neighbor
from app.store import insert_simulation
from dbtools import in_rolled_back_transaction, place, script_question, simulation

RARE = "quizzlewump"  # in one answer
COMMON = "glorptastic"  # in three answers

# Distances from the question, chosen so the vector search alone would not accept a, c or e.
DISTANCES = {"a1": 0.10, "a2": 0.80, "b1": 0.85, "b2": 0.90, "c1": 0.95, "c2": 0.99}
TEXTS = {
    "a1": "A nice calm answer with nothing special.",
    "a2": f"Honestly the {RARE} was the whole problem.",
    "b1": f"I felt {COMMON} about it.",
    "b2": f"It was {COMMON} and also {COMMON}.",
    "c1": f"{COMMON} all the way.",
    "c2": "Another plain answer.",
}


async def between_rare_and_common(conn):
    """A keyword threshold that the rare word clears and the common word does not.

    A word's weight is ln(rows in the table / rows containing it), so it depends on how many real rows exist.
    The rare word (1 row) weighs ln(n) and the common one (3 rows) weighs ln(n) - 1.1, whatever n is.
    """
    return math.log(await conn.fetchval("SELECT count(*) FROM responses")) - 0.5


async def seed(conn):
    """A study whose answers carry the made-up words. Returns (study id, response id by original name)."""
    study_id = await insert_simulation(conn, "b", 0, "m", simulation())
    ids = {}
    for original, text in TEXTS.items():
        ids[original] = await conn.fetchval(
            """
            UPDATE responses SET answer = $1
            WHERE answer = $2 AND respondent_id IN (SELECT id FROM respondents WHERE study_id = $3)
            RETURNING id
            """,
            text, original, study_id,
        )
    # place() finds rows by answer text, so set the vectors by id here instead
    from app.embeddings import to_vector_literal
    from dbtools import at_distance

    for original, distance in DISTANCES.items():
        await conn.execute(
            "UPDATE responses SET embedding = $1::vector, embedding_model = 'm' WHERE id = $2",
            to_vector_literal(at_distance(distance)), ids[original],
        )
    return study_id, ids


# ---- query terms and fusion --------------------------------------------------------------------


def test_keyword_terms_drop_survey_boilerplate_short_words_and_repeats():
    assert keyword_terms("Do respondents mention knee pain?") == ["knee", "pain"]
    # "people" and "say" are survey boilerplate and "is" is too short. Plain stop words like "why" and
    # "the" stay, because Postgres removes those itself when it builds the query.
    assert keyword_terms("Why do people say the app is glitchy, glitchy!") == ["why", "the", "app", "glitchy"]
    assert keyword_terms("Is it OK to go?") == []


def test_fusion_rewards_agreement_between_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 4, 1]])

    assert fused[:2] == [1, 3]  # in both lists, so ahead of the ids that appear once
    assert set(fused) == {1, 2, 3, 4}


def test_fusion_scores_follow_the_formula_and_ties_break_by_id():
    # id 7 is first in one list only, id 3 is first in the other: the same score, so the lower id wins
    assert reciprocal_rank_fusion([[7], [3]]) == [3, 7]
    # 1/(60+1) beats 1/(60+2) beats 1/(60+3)
    assert reciprocal_rank_fusion([[9, 5, 8]]) == [9, 5, 8]


def test_a_smaller_constant_makes_top_ranks_count_for_more():
    lists = [[1, 2], [2, 3, 4, 5, 6, 1]]

    assert reciprocal_rank_fusion(lists, constant=60)[0] == 2
    assert reciprocal_rank_fusion([[1], [2], [2]], constant=1)[0] == 2


# ---- keyword search ----------------------------------------------------------------------------


def test_keyword_search_finds_stemmed_matches_and_ranks_rare_words_higher():
    async def body(conn):
        study_id, ids = await seed(conn)

        found = await keyword_search(conn, f"What about {RARE} and {COMMON}s?", 10, study_id)

        order = [response_id for response_id, _ in found]
        assert set(order) == {ids["a2"], ids["b1"], ids["b2"], ids["c1"]}  # "glorptastics" matched "glorptastic"
        assert order[0] == ids["a2"]  # the one-answer word outweighs the three-answer word
        scores = dict(found)
        assert scores[ids["a2"]] > scores[ids["b1"]] > 0
        assert scores[ids["b1"]] == pytest.approx(scores[ids["c1"]])  # one match each, same word, same weight

    in_rolled_back_transaction(body)


def test_a_response_matching_both_words_beats_one_matching_either():
    async def body(conn):
        study_id, ids = await seed(conn)
        await conn.execute("UPDATE responses SET answer = $1 WHERE id = $2", f"{RARE} and {COMMON} together", ids["c2"])

        found = await keyword_search(conn, f"{RARE} {COMMON}", 10, study_id)

        assert found[0][0] == ids["c2"]

    in_rolled_back_transaction(body)


def test_keyword_search_can_be_limited_to_a_study_and_a_count():
    async def body(conn):
        study_id, ids = await seed(conn)
        other_study, other_ids = await seed(conn)

        scoped = await keyword_search(conn, RARE, 10, study_id)
        everywhere = await keyword_search(conn, RARE, 100, None)
        one = await keyword_search(conn, COMMON, 1, study_id)

        assert [i for i, _ in scoped] == [ids["a2"]]
        assert {ids["a2"], other_ids["a2"]} <= {i for i, _ in everywhere}
        assert len(one) == 1

    in_rolled_back_transaction(body)


def test_a_question_with_no_content_words_or_no_matches_returns_nothing():
    async def body(conn):
        study_id, _ = await seed(conn)

        assert await keyword_search(conn, "Is it OK?", 10, study_id) == []
        assert await keyword_search(conn, "zzzxyznotaword", 10, study_id) == []

    in_rolled_back_transaction(body)


# ---- hybrid retrieval --------------------------------------------------------------------------


def test_hybrid_fuses_both_lists_and_marks_which_hits_the_keywords_found(stub_llm, monkeypatch):
    script_question(stub_llm)

    async def body(conn):
        study_id, ids = await seed(conn)
        monkeypatch.setattr(retrieval, "MIN_KEYWORD_SCORE", await between_rare_and_common(conn))

        found = await hybrid_candidates(conn, f"What about {RARE}?", [1.0] + [0.0] * 1535, "m", study_id, n=6)

        order = [hit.response_id for hit in found.hits]
        assert order[0] == ids["a1"] or order[0] == ids["a2"]
        # a2 is far in meaning (0.80) but is the only keyword match, so fusion lifts it above the other far ones
        assert order.index(ids["a2"]) < order.index(ids["b1"])
        by_id = {hit.response_id: hit for hit in found.hits}
        assert by_id[ids["a2"]].keyword_match is True and by_id[ids["a1"]].keyword_match is False
        assert found.best_distance == pytest.approx(0.10)
        assert by_id[ids["a2"]].distance == pytest.approx(0.80)

    in_rolled_back_transaction(body)


def test_a_keyword_hit_outside_the_vector_top_list_is_still_included_with_its_distance(stub_llm, monkeypatch):

    async def body(conn):
        study_id, ids = await seed(conn)
        monkeypatch.setattr(retrieval, "MIN_KEYWORD_SCORE", await between_rare_and_common(conn))

        found = await hybrid_candidates(conn, f"{RARE}", [1.0] + [0.0] * 1535, "m", study_id, n=2)  # vector list holds only 2

        by_id = {hit.response_id: hit for hit in found.hits}
        assert ids["a2"] in by_id  # not among the 2 closest by vector, but the keyword search found it
        assert by_id[ids["a2"]].distance == pytest.approx(0.80)  # hydrated with its real distance

    in_rolled_back_transaction(body)


# ---- ask: evidence from keywords ----------------------------------------------------------------


def script_answers(stub_llm, answer):
    """Answer for Answer requests and relevance scores (all 3) for Reranking requests."""

    async def parse(model, messages, response_format):
        stub_llm.calls.append(messages)
        if response_format is Reranking:
            ids = [int(i) for i in __import__("re").findall(r"\[R(\d+)\]", messages[1]["content"])]
            parsed = Reranking(scores=[Relevance(response_id=i, score=3) for i in ids])
        else:
            parsed = answer
        return NS(
            usage=NS(prompt_tokens=1, completion_tokens=1),
            choices=[NS(message=NS(refusal=None, parsed=parsed))],
        )

    stub_llm.parse = parse


def a_citation(response_id):
    return Answer(answerable=True, answer="Yes.", citations=[Citation(response_id=response_id, quote="x")])


def test_a_distant_response_with_a_distinctive_keyword_becomes_evidence(stub_llm, monkeypatch):
    script_question(stub_llm)

    async def body(conn):
        study_id, ids = await seed(conn)
        monkeypatch.setattr(retrieval, "MIN_KEYWORD_SCORE", await between_rare_and_common(conn))
        script_answers(stub_llm, a_citation(ids["a2"]))

        response = await ask(conn, f"Did anyone mention {RARE}?", study_id, embedding_model="m", max_distance=0.5)

        user = stub_llm.calls[-1][1]["content"]
        assert response.status == "answered"
        assert f"[R{ids['a2']}]" in user  # distance 0.80 is past the 0.5 threshold, but the keyword qualified it
        assert f"[R{ids['b1']}]" not in user  # distance 0.85 and no keyword match

    in_rolled_back_transaction(body)


def test_vector_mode_ignores_keywords(stub_llm):
    script_question(stub_llm)

    async def body(conn):
        study_id, ids = await seed(conn)

        response = await ask(
            conn, f"Did anyone mention {RARE}?", study_id, embedding_model="m", max_distance=0.05, mode="vector"
        )

        assert response.reason == "retrieval_too_weak" and stub_llm.calls == []  # a1 at 0.10 is too far, and keywords are off

    in_rolled_back_transaction(body)


def test_a_common_word_alone_is_not_enough_evidence(stub_llm, monkeypatch):
    script_question(stub_llm)

    async def body(conn):
        study_id, _ = await seed(conn)
        monkeypatch.setattr(retrieval, "MIN_KEYWORD_SCORE", await between_rare_and_common(conn))

        response = await ask(conn, f"Is it {COMMON}?", study_id, embedding_model="m", max_distance=0.05)

        assert response.reason == "retrieval_too_weak" and stub_llm.calls == []

    in_rolled_back_transaction(body)


def test_nothing_close_and_no_keyword_is_refused_without_calling_the_model(stub_llm):
    script_question(stub_llm)

    async def body(conn):
        study_id, _ = await seed(conn)

        response = await ask(conn, "What is the capital of France?", study_id, embedding_model="m", max_distance=0.05)

        assert response.status == "not_enough_evidence" and stub_llm.calls == []

    in_rolled_back_transaction(body)


def test_rerank_in_ask_scores_the_candidates_before_answering(stub_llm, monkeypatch):
    script_question(stub_llm)

    async def body(conn):
        study_id, ids = await seed(conn)
        monkeypatch.setattr(retrieval, "MIN_KEYWORD_SCORE", await between_rare_and_common(conn))
        script_answers(stub_llm, a_citation(ids["a2"]))

        await ask(conn, f"Did anyone mention {RARE}?", study_id, embedding_model="m", max_distance=0.5, rerank=True)

        assert len(stub_llm.calls) == 2  # one call to score the candidates, then one to answer
        assert "Answers:" in stub_llm.calls[0][1]["content"] and "Evidence:" in stub_llm.calls[1][1]["content"]

    in_rolled_back_transaction(body)


# ---- rerank on its own -------------------------------------------------------------------------


def hits(*ids):
    return [Neighbor(0.5, 1, "b", 1, f"answer {i}", None, response_id=i, question="q") for i in ids]


def script_scores(stub_llm, scores):
    async def parse(model, messages, response_format):
        stub_llm.calls.append(messages)
        parsed = Reranking(scores=[Relevance(response_id=i, score=s) for i, s in scores.items()])
        return NS(
            usage=NS(prompt_tokens=1, completion_tokens=1),
            choices=[NS(message=NS(refusal=None, parsed=parsed))],
        )

    stub_llm.parse = parse


def test_rerank_orders_by_score_keeps_ties_in_their_old_order_and_drops_zeros(stub_llm):
    script_scores(stub_llm, {1: 1, 2: 3, 3: 0, 4: 3, 5: 2})

    result = asyncio.run(rerank("q", hits(1, 2, 3, 4, 5), top_n=10))

    assert [h.response_id for h in result] == [2, 4, 5, 1]  # 3s in their old order, then 2, then 1; the 0 is gone


def test_rerank_keeps_only_top_n(stub_llm):
    script_scores(stub_llm, {1: 3, 2: 3, 3: 3})

    assert [h.response_id for h in asyncio.run(rerank("q", hits(1, 2, 3), top_n=2))] == [1, 2]


def test_candidates_the_model_forgot_to_score_count_as_zero_and_wild_scores_are_clamped(stub_llm):
    script_scores(stub_llm, {1: 99, 2: -4})  # id 3 is missing

    result = asyncio.run(rerank("q", hits(1, 2, 3), top_n=5))

    assert [h.response_id for h in result] == [1]  # 99 became 3, -4 became 0, the missing one is 0


def test_the_rerank_prompt_lists_every_candidate_with_its_id(stub_llm):
    script_scores(stub_llm, {7: 3, 9: 3})

    asyncio.run(rerank("Does it hurt?", hits(7, 9), top_n=5))

    user = stub_llm.calls[0][1]["content"]
    assert user.startswith("Question: Does it hurt?") and "[R7]" in user and "[R9]" in user


def test_rerank_with_no_candidates_makes_no_call(stub_llm):
    assert asyncio.run(rerank("q", [], top_n=5)) == [] and stub_llm.calls == []


# ---- the measurement ---------------------------------------------------------------------------


def test_precision_counts_relevant_answers_in_the_top_k_and_does_not_reward_returning_fewer():
    found = hits(1, 2, 3)
    found[0].answer, found[1].answer, found[2].answer = "knee pain", "nothing", "my KNEES hurt"

    assert precision_at_k(found, r"\bknees?\b") == pytest.approx(2 / 5)  # 2 relevant out of a possible 5
    assert precision_at_k(found, r"\bknees?\b", k=3) == pytest.approx(2 / 3)
    assert precision_at_k([], r"x") == 0.0


def test_the_report_averages_and_counts_questions_with_a_relevant_result():
    report = EvalReport(
        [
            QuestionResult("Q one?", 10, {"vector": 0.8, "keyword": 0.0}, {"vector": 5, "keyword": 0}),
            QuestionResult("Q two?", 4, {"vector": 0.4, "keyword": 1.0}, {"vector": 5, "keyword": 5}),
        ]
    )

    assert report.mean_precision("vector") == pytest.approx(0.6)
    assert report.found_something("keyword") == 1
    text = format_eval(report)
    assert "Q one?" in text and "0.60" in text and "1/2" in text
