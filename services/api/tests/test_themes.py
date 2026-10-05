import re
from types import SimpleNamespace as NS

import numpy as np
import pytest
from pydantic import ValidationError

from app.embeddings import to_vector_literal
from app.respondents import Simulation
from app.schemas import GroupLabel, GroupLabels
from app.store import insert_simulation
from app.themes import build_run, format_run, label_themes, save_run
from dbtools import DIMS, in_rolled_back_transaction, persona


def vector(axis, rng, noise=0.03):
    """A 1536-dimension vector near one axis."""
    v = np.zeros(DIMS)
    v[axis] = 1.0
    v[200:260] += rng.normal(0, noise, 60)
    return v.tolist()


async def seed(conn, blob_sizes=(8, 8, 8), far=0, model="m"):
    """A study with one question. Answers 'blobN ...' are near axis N, and 'stray ...' answers are near none."""
    rng = np.random.RandomState(0)
    answers, vectors = [], []
    for blob, size in enumerate(blob_sizes):
        for n in range(size):
            answers.append(f"blob{blob} answer {n}")
            vectors.append(vector(blob, rng))
    for n in range(far):
        answers.append(f"stray answer {n}")
        vectors.append(vector(900 + n, rng))
    sim = Simulation(
        questions=["Why did you stop?"],
        personas=[persona(f"P{i}", 30) for i in range(len(answers))],
        answers=[[a] for a in answers],
    )
    study_id = await insert_simulation(conn, "a meal-kit study", 0, "m", sim)
    for answer, v in zip(answers, vectors):
        await conn.execute(
            "UPDATE responses SET embedding = $1::vector, embedding_model = $2 WHERE answer = $3 "
            "AND respondent_id IN (SELECT id FROM respondents WHERE study_id = $4)",
            to_vector_literal(v), model, answer, study_id,
        )
    return study_id


def script_labels(stub_llm, duplicate=False, skip=None):
    """Name each group after the blob its first answer came from, so tests can check the mapping."""

    async def parse(model, messages, response_format):
        stub_llm.calls.append(messages)
        entries = []
        for number, text in enumerate(re.findall(r"Group (\d+):\n((?:- .*\n?)+)", messages[1]["content"]), 1):
            first = text[1].splitlines()[0]
            name = re.search(r"(blob\d|stray)", first).group(1)
            entries.append(GroupLabel(group=int(text[0]), label="Same label" if duplicate else f"About {name}", description=f"Answers from {name}"))
        if skip:
            entries = [e for e in entries if e.group != skip]
        return NS(
            usage=NS(prompt_tokens=1, completion_tokens=1),
            choices=[NS(message=NS(refusal=None, parsed=GroupLabels.model_construct(groups=entries)))],
        )

    stub_llm.parse = parse


def test_each_cluster_is_named_from_its_own_answers_and_themes_are_counted(stub_llm, ledger):
    script_labels(stub_llm)

    async def body(conn):
        study_id = await seed(conn)

        run = await build_run(conn, study_id, 1, k=3, model="m")

        assert [t.kind for t in run.themes] == ["theme"] * 3
        assert sorted(len(t.response_ids) for t in run.themes) == [8, 8, 8]
        for theme in run.themes:
            blob = theme.label.removeprefix("About ")
            answers = [run.answers[i] for i in theme.response_ids]
            assert all(a.startswith(blob) for a in answers)  # the label matches what is inside the theme
        assert run.n_responses == 24 and run.question == "Why did you stop?" and run.method == "kmeans"
        assert run.silhouette > 0.5 and run.params["k"] == 3
        assert len(stub_llm.calls) == 1  # all groups were named in one call
        assert ledger[0]["prompt"].name == "theme_label" and ledger[0]["prompt"].version == "v2"

    in_rolled_back_transaction(body)


def test_the_naming_prompt_shows_the_brief_the_question_and_every_group(stub_llm):
    script_labels(stub_llm)

    async def body(conn):
        study_id = await seed(conn)

        await build_run(conn, study_id, 1, k=3, model="m")

        user = stub_llm.calls[0][1]["content"]
        assert user.startswith("Study brief: a meal-kit study\nQuestion asked: Why did you stop?")
        assert all(f"Group {n}:" in user for n in (1, 2, 3))
        assert user.count("- blob") == 15  # five of the closest answers per group

    in_rolled_back_transaction(body)


def test_responses_that_fit_no_theme_go_to_the_outlier_bucket(stub_llm):
    script_labels(stub_llm)

    async def body(conn):
        study_id = await seed(conn, far=2)

        run = await build_run(conn, study_id, 1, k=4, model="m")

        outliers = [t for t in run.themes if t.kind == "outlier"]
        assert len(outliers) == 1 and outliers[0].label == "Outliers"
        assert all(run.answers[i].startswith("stray") for i in outliers[0].response_ids)
        assert run.themes[-1].kind == "outlier"  # always listed last
        assert sum(len(t.response_ids) for t in run.themes) == run.n_responses  # nobody is lost or counted twice

    in_rolled_back_transaction(body)


def test_a_group_the_model_forgot_to_name_gets_a_placeholder(stub_llm):
    script_labels(stub_llm, skip=2)

    async def body(conn):
        study_id = await seed(conn)

        run = await build_run(conn, study_id, 1, k=3, model="m")

        assert sorted(t.label for t in run.themes if t.label.startswith("Theme")) == ["Theme 2"]
        assert len(run.themes) == 3

    in_rolled_back_transaction(body)


def test_too_few_answers_is_a_clear_error(stub_llm):
    async def body(conn):
        study_id = await seed(conn, blob_sizes=(3,))

        with pytest.raises(ValueError, match="at least 8 are needed"):
            await build_run(conn, study_id, 1, k=2, model="m")

    in_rolled_back_transaction(body)


def test_only_answers_with_vectors_from_the_requested_model_count(stub_llm):
    async def body(conn):
        study_id = await seed(conn, model="other-model")

        with pytest.raises(ValueError, match="0 embedded answers"):
            await build_run(conn, study_id, 1, k=3, model="m")

    in_rolled_back_transaction(body)


def test_hdbscan_runs_through_the_same_pipeline(stub_llm):
    script_labels(stub_llm)

    async def body(conn):
        study_id = await seed(conn, blob_sizes=(10, 10, 10))

        run = await build_run(conn, study_id, 1, method="hdbscan", min_cluster_size=5, model="m")

        assert run.method == "hdbscan" and run.params["min_cluster_size"] == 5
        assert len([t for t in run.themes if t.kind == "theme"]) == 3

    in_rolled_back_transaction(body)


def test_labels_must_be_different_from_each_other():
    same = [GroupLabel(group=1, label="Price", description="a"), GroupLabel(group=2, label=" price ", description="b")]

    with pytest.raises(ValidationError, match="share one"):
        GroupLabels(groups=same)
    assert len(GroupLabels(groups=[same[0], GroupLabel(group=2, label="Delivery", description="c")]).groups) == 2


def test_label_themes_returns_labels_by_group_number(stub_llm):
    script_labels(stub_llm)

    import asyncio

    named = asyncio.run(label_themes("b", "q", [["blob0 x"], ["blob1 y"]], "m", [[1], [2]]))

    assert named == {1: ("About blob0", "Answers from blob0"), 2: ("About blob1", "Answers from blob1")}


def test_a_run_is_stored_with_its_themes_members_and_counts(stub_llm):
    script_labels(stub_llm)

    async def body(conn):
        study_id = await seed(conn, far=2)
        run = await build_run(conn, study_id, 1, k=4, model="m")

        run_id = await save_run(conn, run)

        saved = await conn.fetchrow("SELECT * FROM theme_runs WHERE id = $1", run_id)
        assert (saved["study_id"], saved["question_number"], saved["method"], saved["n_responses"]) == (study_id, 1, "kmeans", 26)
        stored = await conn.fetch("SELECT * FROM themes WHERE run_id = $1 ORDER BY size DESC, id", run_id)
        assert {t["kind"] for t in stored} == {"theme", "outlier"}
        for row in stored:
            members = await conn.fetch("SELECT * FROM theme_members WHERE theme_id = $1", row["id"])
            assert len(members) == row["size"]  # the stored count matches the stored members
            assert all(m["distance"] >= 0 for m in members)
        assert sum(t["size"] for t in stored) == 26
        everyone = await conn.fetchval(
            "SELECT count(DISTINCT response_id) FROM theme_members WHERE theme_id = ANY($1)", [t["id"] for t in stored]
        )
        assert everyone == 26  # every response is in exactly one theme or the outlier bucket

    in_rolled_back_transaction(body)


def test_deleting_a_study_removes_its_theme_runs(stub_llm):
    script_labels(stub_llm)

    async def body(conn):
        study_id = await seed(conn)
        run_id = await save_run(conn, await build_run(conn, study_id, 1, k=3, model="m"))

        await conn.execute("DELETE FROM studies WHERE id = $1", study_id)

        assert await conn.fetchval("SELECT count(*) FROM theme_runs WHERE id = $1", run_id) == 0
        assert await conn.fetchval("SELECT count(*) FROM themes WHERE run_id = $1", run_id) == 0

    in_rolled_back_transaction(body)


def test_the_report_shows_counts_shares_examples_and_outliers(stub_llm):
    script_labels(stub_llm)

    async def body(conn):
        study_id = await seed(conn, far=2)
        run = await build_run(conn, study_id, 1, k=4, model="m")

        text = format_run(run)

        assert f'study {study_id} Q1: "Why did you stop?"' in text
        assert "kmeans, 26 responses, silhouette" in text
        assert "1. About blob" in text and "(8, 31%)" in text
        assert re.search(r'\[R\d+\] "blob\d answer \d"', text)  # example quotes
        assert "Outliers (2): fit none of the themes" in text

    in_rolled_back_transaction(body)
