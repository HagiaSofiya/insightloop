import asyncio
from types import SimpleNamespace as NS

import pytest

from app import chunk_experiment as experiment
from app.chunk_experiment import (
    Hit, build_chunks, embed_chunks, format_experiment, mention_rate, run_experiment, top_responses,
)
from app.chunking import sentence_chunks, split_sentences, window_chunks
from app.store import insert_simulation
from dbtools import in_rolled_back_transaction, simulation

DIMS = 1536


def test_sentences_split_on_end_punctuation_followed_by_a_capital():
    text = "It was fine. Really? Yes! But the box (e.g. the big one) was late. then it arrived."

    assert split_sentences(text) == [
        "It was fine.", "Really?", "Yes!", "But the box (e.g. the big one) was late. then it arrived.",
    ]  # the lowercase after "late." is not treated as a new sentence
    assert split_sentences("  ") == [] and split_sentences("One only") == ["One only"]


def test_a_sentence_chunk_is_one_sentence():
    assert sentence_chunks("A one. B two. C three.") == ["A one.", "B two.", "C three."]


def sentences(n):
    return " ".join(f"S{i}." for i in range(1, n + 1))


def test_windows_overlap_by_one_sentence_and_cover_everything():
    assert window_chunks(sentences(7)) == ["S1. S2. S3.", "S3. S4. S5.", "S5. S6. S7."]


def test_the_last_window_is_added_when_the_stride_would_miss_the_end():
    windows = window_chunks(sentences(8))

    assert windows[-1] == "S6. S7. S8."
    assert all(f"S{i}." in " ".join(windows) for i in range(1, 9))


def test_short_answers_make_one_window_and_empty_text_makes_none():
    assert window_chunks(sentences(3)) == ["S1. S2. S3."]
    assert window_chunks(sentences(1)) == ["S1."]
    assert window_chunks("") == []


def test_mention_rate_checks_the_whole_answer_not_the_matched_chunk():
    hits = [Hit(0.1, "a chunk", "this answer says TRUST issues"), Hit(0.2, "b", "nothing here")]

    assert mention_rate(hits, r"trust") == 0.5
    assert mention_rate([], r"trust") == 0.0


ANSWER = "The app looks nice. I like the colors. Honestly I do not trust it with my data. Support was fine."
TWO_TRUST = "The box came late. Honestly I do not trust it with my data. I really do not trust their app. Support was fine."
OTHER = "Great price. Fast delivery. Nice box. Would order again."


def meaning(text):
    """A tiny stand-in for a real embedding. Dimension 0 is 'trust', dimension 1 is 'everything else'."""
    parts = [p for p in text.replace("?", ".").split(".") if p.strip()]
    trust = sum("trust" in p for p in parts) / len(parts)
    vector = [0.0] * DIMS
    vector[0], vector[1] = trust, 1 - trust
    return vector


def script_meaning(stub_llm):
    async def embed_create(model, input):
        return NS(data=[NS(index=i, embedding=meaning(t)) for i, t in enumerate(input)], usage=NS(prompt_tokens=1))

    stub_llm.embed_create = embed_create


async def seeded_study(conn):
    study_id = await insert_simulation(conn, "b", 0, "m", simulation())
    await conn.execute(
        """
        UPDATE responses SET answer = CASE answer WHEN 'a1' THEN $1 WHEN 'b1' THEN $2 ELSE $3 END
        WHERE respondent_id IN (SELECT id FROM respondents WHERE study_id = $4)
        """,
        ANSWER, TWO_TRUST, OTHER, study_id,
    )
    return study_id


def test_chunks_are_built_once_and_embedded_through_the_cache(stub_llm):
    script_meaning(stub_llm)

    async def body(conn):
        study_id = await seeded_study(conn)

        added = await build_chunks(conn, "sentence", study_id)
        again = await build_chunks(conn, "sentence", study_id)
        first = await embed_chunks(conn, "sentence", "m", study_id)
        second = await embed_chunks(conn, "sentence", "m", study_id)

        assert added == 24 and again == 0  # four sentences in each of the six answers
        distinct = set(sentence_chunks(ANSWER) + sentence_chunks(TWO_TRUST) + sentence_chunks(OTHER))
        assert first.embedded == 24 and first.api_texts == len(distinct)
        assert second.embedded == 0 and second.requests == 0

    in_rolled_back_transaction(body)


def test_a_chunk_finds_an_idea_that_the_whole_answer_dilutes(stub_llm):
    script_meaning(stub_llm)
    query = meaning("I do not trust it.")  # all 'trust'

    async def body(conn):
        study_id = await seeded_study(conn)
        await run_experiment(conn, "m", study_id)

        whole = await top_responses(conn, "whole", query, "m", k=10)
        sentence = await top_responses(conn, "sentence", query, "m", k=10)

        whole_by_answer = {hit.answer: hit for hit in whole}
        sentence_by_answer = {hit.answer: hit for hit in sentence}
        # One trust sentence in four dilutes the whole-answer vector, but the sentence itself matches exactly.
        assert whole_by_answer[ANSWER].distance > 0.5
        assert sentence_by_answer[ANSWER].distance == pytest.approx(0)
        assert sentence_by_answer[ANSWER].text == "Honestly I do not trust it with my data."
        # A response with two matching sentences still counts once.
        assert [hit.answer for hit in sentence].count(TWO_TRUST) == 1
        assert len(sentence) == len(whole) == 6  # six responses, however many chunks each has

    in_rolled_back_transaction(body)


def test_the_experiment_report_compares_the_three_strategies(stub_llm, monkeypatch):
    script_meaning(stub_llm)
    monkeypatch.setattr(experiment, "CONCEPTS", [("I do not trust it", r"trust"), ("great price", r"price")])

    async def body(conn):
        study_id = await seeded_study(conn)

        report = await run_experiment(conn, "m", study_id)
        text = format_experiment(report)

        assert set(report.chunk_counts) == {"sentence", "window"}
        assert report.chunk_counts["sentence"] == 24 and report.chunk_counts["window"] == 12  # 2 windows per answer
        trust = report.concepts[0]
        assert trust.mention_rate["sentence"] == pytest.approx(0.4)  # 2 of the 5 best responses mention trust
        assert set(trust.best) == {"whole", "sentence", "window"}
        assert trust.best["sentence"].distance < trust.best["whole"].distance  # the sentence is the closer match
        assert report.avg_chars["sentence"] < report.avg_chars["whole"]  # chunks are shorter than answers
        assert "whole" in text and "sentence" in text and "window" in text and "mean" in text

    in_rolled_back_transaction(body)
