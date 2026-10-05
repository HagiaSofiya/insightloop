"""The golden dataset: a small fake study whose true answers are known, and the cases that test against it.

Each respondent in evals/golden_dataset.json carries a tag per question saying which topics their answer
mentions. Those tags are the ground truth: a case's relevant responses and true counts come from them.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import asyncpg

from app.embeddings import EMBEDDING_MODEL, embed_responses
from app.importer import ImportedData
from app.store import insert_import

EVALS_DIR = Path(__file__).resolve().parents[3] / "evals"
DATASET_PATH = EVALS_DIR / "golden_dataset.json"
CASES_PATH = EVALS_DIR / "cases.json"


@dataclass
class Golden:
    brief: str
    questions: list[str]
    respondents: list[dict]  # name, age, gender, answers (one per question), tags (one list per question)

    def tagged(self, question_number: int, topic: str, max_age: int | None = None) -> list[dict]:
        """The respondents whose answer to this question mentions the topic (optionally only those under max_age)."""
        return [
            r for r in self.respondents
            if topic in r["tags"][question_number - 1] and (max_age is None or r["age"] < max_age)
        ]


def load_golden(path: Path = DATASET_PATH) -> Golden:
    data = json.loads(path.read_text())
    return Golden(data["brief"], data["questions"], data["respondents"])


def load_cases(path: Path = CASES_PATH) -> list[dict]:
    return json.loads(path.read_text())["cases"]


async def seed_golden(conn: asyncpg.Connection, golden: Golden, model: str = EMBEDDING_MODEL) -> int:
    """Make sure the golden study exists and is embedded. Returns its study id. Safe to run repeatedly."""
    study_id = await conn.fetchval("SELECT id FROM studies WHERE brief = $1 ORDER BY id LIMIT 1", golden.brief)
    if study_id is None:
        data = ImportedData(
            golden.questions,
            [{k: r[k] for k in ("name", "age", "gender")} for r in golden.respondents],
            [r["answers"] for r in golden.respondents],
        )
        study_id = await insert_import(conn, golden.brief, data)
    await embed_responses(conn, model, study_id)
    return study_id


async def relevant_ids(conn: asyncpg.Connection, golden: Golden, study_id: int, case: dict) -> set[int]:
    """Database ids of the responses a case counts as relevant: tagged with its topic, or with a topic it lists as also relevant
    (a price question is also answered by "a cheaper plan would bring me back" in the next question)."""
    ids: set[int] = set()
    for where in [case, *case.get("also_relevant", [])]:
        names = [r["name"] for r in golden.tagged(where["question_number"], where["topic"])]
        rows = await conn.fetch(
            """
            SELECT s.id FROM responses s JOIN respondents r ON r.id = s.respondent_id
            WHERE r.study_id = $1 AND s.question_number = $2 AND r.name = ANY($3)
            """,
            study_id, where["question_number"], names,
        )
        ids |= {row["id"] for row in rows}
    return ids


async def ensure_themes(conn: asyncpg.Connection, study_id: int, questions: int) -> int:
    """Cluster and label each question of the golden study that has no theme run yet. Returns how many were built.

    The count cases go through the agent, and the agent's theme tools need a run to read.
    """
    from app.themes import build_run, save_run  # imported here: it pulls in scikit-learn and the LLM client

    built = 0
    for number in range(1, questions + 1):
        if await conn.fetchval("SELECT 1 FROM theme_runs WHERE study_id = $1 AND question_number = $2", study_id, number):
            continue
        await save_run(conn, await build_run(conn, study_id, number))
        built += 1
    return built
