"""Tests that run real SQL against Postgres, inside a transaction that is always rolled back."""

from pathlib import Path

import asyncpg
import pytest

from app import store
from app.importer import read_csv
from app.store import insert_import, insert_simulation
from dbtools import in_rolled_back_transaction, simulation

SAMPLE = Path(__file__).parents[3] / "evals" / "sample_responses.csv"


def test_a_simulation_is_saved_as_a_study_with_respondents_and_responses():
    async def body(conn):
        study_id = await insert_simulation(conn, "a brief", 7, "gpt-4o-mini", simulation())

        study = await conn.fetchrow("SELECT * FROM studies WHERE id = $1", study_id)
        assert (study["brief"], study["seed"], study["model"]) == ("a brief", 7, "gpt-4o-mini")
        respondents = await conn.fetch("SELECT * FROM respondents WHERE study_id = $1 ORDER BY id", study_id)
        assert [(r["name"], r["age"], r["source"]) for r in respondents] == [
            ("Ann", 25, "synthetic"), ("Bo", 40, "synthetic"), ("Cy", 61, "synthetic"),
        ]
        rows = await conn.fetch(
            """
            SELECT r.name, s.question_number, s.question, s.answer
            FROM responses s JOIN respondents r ON r.id = s.respondent_id
            WHERE r.study_id = $1 ORDER BY r.id, s.question_number
            """,
            study_id,
        )
        assert len(rows) == 6  # 3 respondents x 2 questions
        assert (rows[0]["name"], rows[0]["question_number"], rows[0]["question"], rows[0]["answer"]) == (
            "Ann", 1, "Why try it?", "a1",
        )
        assert rows[-1]["answer"] == "c2"

    in_rolled_back_transaction(body)


def test_typed_columns_make_filtering_by_age_plain_sql():
    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())

        young = await conn.fetch(
            """
            SELECT r.name, s.answer FROM respondents r JOIN responses s ON s.respondent_id = r.id
            WHERE r.study_id = $1 AND r.age < 45 AND s.question_number = 2 ORDER BY r.age
            """,
            study_id,
        )

        assert [(row["name"], row["answer"]) for row in young] == [("Ann", "a2"), ("Bo", "b2")]

    in_rolled_back_transaction(body)


def test_the_database_enforces_its_own_rules():
    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())
        respondent_id = await conn.fetchval("SELECT id FROM respondents WHERE study_id = $1 LIMIT 1", study_id)

        async def rejected(sql, *args, error):
            with pytest.raises(error):
                async with conn.transaction():  # a savepoint, so the failure does not abort the test
                    await conn.execute(sql, *args)

        await rejected(  # the same question answered twice by one respondent
            "INSERT INTO responses (respondent_id, question_number, question, answer) VALUES ($1, 1, 'q', 'a')",
            respondent_id, error=asyncpg.UniqueViolationError,
        )
        await rejected(  # an unknown source
            "INSERT INTO respondents (study_id, source) VALUES ($1, 'robot')",
            study_id, error=asyncpg.CheckViolationError,
        )
        await rejected(  # an impossible age
            "INSERT INTO respondents (study_id, source, age) VALUES ($1, 'imported', 400)",
            study_id, error=asyncpg.CheckViolationError,
        )
        await rejected(  # a respondent that belongs to no study
            "INSERT INTO respondents (study_id, source) VALUES (999999999, 'imported')",
            error=asyncpg.ForeignKeyViolationError,
        )

    in_rolled_back_transaction(body)


def test_deleting_a_study_removes_its_respondents_and_responses():
    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())
        respondent_ids = [
            r["id"] for r in await conn.fetch("SELECT id FROM respondents WHERE study_id = $1", study_id)
        ]
        assert len(respondent_ids) == 3
        assert await conn.fetchval(
            "SELECT count(*) FROM responses WHERE respondent_id = ANY($1)", respondent_ids
        ) == 6

        await conn.execute("DELETE FROM studies WHERE id = $1", study_id)

        assert await conn.fetchval("SELECT count(*) FROM respondents WHERE study_id = $1", study_id) == 0
        assert await conn.fetchval(
            "SELECT count(*) FROM responses WHERE respondent_id = ANY($1)", respondent_ids
        ) == 0

    in_rolled_back_transaction(body)


def test_save_simulation_opens_a_connection_saves_and_closes_it(monkeypatch):
    class KeepOpen:
        """Hands the test's transaction to the code under test, and records the close."""

        def __init__(self, conn):
            self._conn = conn
            self.closed = False

        def __getattr__(self, name):
            return getattr(self._conn, name)

        async def close(self):
            self.closed = True

    async def body(conn):
        wrapper = KeepOpen(conn)

        async def fake_connect():
            return wrapper

        monkeypatch.setattr(store, "connect", fake_connect)

        study_id = await store.save_simulation("via wrapper", 3, "m", simulation())

        assert wrapper.closed
        assert await conn.fetchval("SELECT brief FROM studies WHERE id = $1", study_id) == "via wrapper"

    in_rolled_back_transaction(body)


def test_an_import_is_stored_as_imported_with_unknown_attributes_null_and_blanks_skipped():
    data = read_csv(str(SAMPLE))

    async def body(conn):
        study_id = await insert_import(conn, "meal kits", data)

        study = await conn.fetchrow("SELECT * FROM studies WHERE id = $1", study_id)
        assert (study["brief"], study["seed"], study["model"]) == ("meal kits", None, None)
        respondents = await conn.fetch("SELECT * FROM respondents WHERE study_id = $1 ORDER BY id", study_id)
        assert {r["source"] for r in respondents} == {"imported"}
        assert [(r["age"], r["gender"], r["name"]) for r in respondents] == [
            (34, "woman", None), (58, "man", None), (27, "non-binary", None), (None, "woman", None),
        ]  # no name column, so name is NULL, and the blank age is NULL
        count = await conn.fetchval(
            "SELECT count(*) FROM responses s JOIN respondents r ON r.id = s.respondent_id WHERE r.study_id = $1",
            study_id,
        )
        assert count == 4 * 2 - 1  # one respondent left the second question blank
        skipped = await conn.fetchval(
            """
            SELECT count(*) FROM responses s JOIN respondents r ON r.id = s.respondent_id
            WHERE r.study_id = $1 AND r.age = 27 AND s.question_number = 2
            """,
            study_id,
        )
        assert skipped == 0

    in_rolled_back_transaction(body)


def test_save_import_goes_through_the_same_connection_wrapper(monkeypatch):
    class KeepOpen:
        def __init__(self, conn):
            self._conn = conn
            self.closed = False

        def __getattr__(self, name):
            return getattr(self._conn, name)

        async def close(self):
            self.closed = True

    async def body(conn):
        wrapper = KeepOpen(conn)

        async def fake_connect():
            return wrapper

        monkeypatch.setattr(store, "connect", fake_connect)

        study_id = await store.save_import("via wrapper", read_csv(str(SAMPLE)))

        assert wrapper.closed
        assert await conn.fetchval("SELECT count(*) FROM respondents WHERE study_id = $1", study_id) == 4

    in_rolled_back_transaction(body)


def test_a_failure_part_way_saves_nothing():
    async def body(conn):
        counts = "SELECT (SELECT count(*) FROM studies), (SELECT count(*) FROM respondents), (SELECT count(*) FROM responses)"
        before = tuple(await conn.fetchrow(counts))
        broken = simulation()
        broken.answers[2] = ["only one answer"]  # the third respondent is missing an answer

        with pytest.raises(ValueError):
            await insert_simulation(conn, "broken", 0, "m", broken)

        assert tuple(await conn.fetchrow(counts)) == before  # nothing from the first two respondents either

    in_rolled_back_transaction(body)
