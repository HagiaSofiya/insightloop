import asyncpg

from app.db import connect
from app.importer import ImportedData
from app.respondents import Simulation


async def _insert_study(
    conn: asyncpg.Connection,
    brief: str,
    seed: int | None,
    model: str | None,
    source: str,
    profiles: list[dict],
    questions: list[str],
    answers: list[list[str | None]],
) -> int:
    """Write a study, its respondents and their responses in one transaction. Returns the study id.

    A profile may lack any attribute (it is stored as NULL) and a blank answer is not stored.
    """
    async with conn.transaction():  # all or nothing
        study_id = await conn.fetchval(
            "INSERT INTO studies (brief, seed, model) VALUES ($1, $2, $3) RETURNING id",
            brief, seed, model,
        )
        for profile, person_answers in zip(profiles, answers, strict=True):
            respondent_id = await conn.fetchval(
                """
                INSERT INTO respondents (study_id, source, name, age, gender, city, income,
                                         household, occupation, backstory, speaking_style)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
                """,
                study_id, source,
                profile.get("name"), profile.get("age"), profile.get("gender"), profile.get("city"),
                profile.get("income"), profile.get("household"), profile.get("occupation"),
                profile.get("backstory"), profile.get("speaking_style"),
            )
            await conn.executemany(
                """
                INSERT INTO responses (respondent_id, question_number, question, answer)
                VALUES ($1, $2, $3, $4)
                """,
                [
                    (respondent_id, number, question, answer)
                    for number, (question, answer) in enumerate(
                        zip(questions, person_answers, strict=True), start=1
                    )
                    if answer
                ],
            )
    return study_id


async def insert_simulation(
    conn: asyncpg.Connection, brief: str, seed: int, model: str, sim: Simulation
) -> int:
    profiles = [persona.model_dump() for persona in sim.personas]
    return await _insert_study(
        conn, brief, seed, model, "synthetic", profiles, sim.questions, sim.answers
    )


async def insert_import(conn: asyncpg.Connection, brief: str, data: ImportedData) -> int:
    return await _insert_study(
        conn, brief, None, None, "imported", data.profiles, data.questions, data.answers
    )


async def save_simulation(brief: str, seed: int, model: str, sim: Simulation) -> int:
    conn = await connect()
    try:
        return await insert_simulation(conn, brief, seed, model, sim)
    finally:
        await conn.close()


async def save_import(brief: str, data: ImportedData) -> int:
    conn = await connect()
    try:
        return await insert_import(conn, brief, data)
    finally:
        await conn.close()
