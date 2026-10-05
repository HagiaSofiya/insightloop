"""Shared helpers for tests that run real SQL against Postgres."""

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace as NS

import asyncpg
import pytest

from app.db import connect
from app.embeddings import to_vector_literal
from app.respondents import Simulation
from app.schemas import Persona

SCHEMA = (Path(__file__).parent.parent / "schema.sql").read_text()


def persona(name, age):
    return Persona(
        name=name, occupation="nurse", backstory="b", speaking_style="terse",
        age=age, gender="woman", city="Austin, TX", income="under $35k", household="lives alone",
    )


def simulation():
    return Simulation(
        questions=["Why try it?", "Why stop?"],
        personas=[persona("Ann", 25), persona("Bo", 40), persona("Cy", 61)],
        answers=[["a1", "a2"], ["b1", "b2"], ["c1", "c2"]],
    )


def in_rolled_back_transaction(body):
    """Run `body(conn)` on a fresh connection, then roll everything back."""

    async def run():
        try:
            conn = await connect()
        except (OSError, asyncpg.PostgresError):
            pytest.skip("Postgres is not running (start it with make up)")
        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute(SCHEMA)  # a no-op if the tables exist, and rolled back if not
            await body(conn)
        finally:
            await transaction.rollback()
            await conn.close()

    asyncio.run(run())


DIMS = 1536


def at_distance(distance):
    """A vector whose cosine distance from the question vector [1, 0, ...] is exactly `distance`."""
    angle = math.acos(1 - distance)
    vector = [0.0] * DIMS
    vector[0], vector[1] = math.cos(angle), math.sin(angle)
    return vector


def script_question(stub_llm):
    """Make the question embed to [1, 0, ...]."""

    async def embed_create(model, input):
        vector = [1.0] + [0.0] * (DIMS - 1)
        return NS(data=[NS(index=0, embedding=vector)], usage=NS(prompt_tokens=1))

    stub_llm.embed_create = embed_create


async def place(conn, study_id, distances):
    """Give each answer in the study a stored vector at the stated distance from the question."""
    ids = {}
    for answer, distance in distances.items():
        ids[answer] = await conn.fetchval(
            """
            UPDATE responses SET embedding = $1::vector, embedding_model = 'm'
            WHERE answer = $2 AND respondent_id IN (SELECT id FROM respondents WHERE study_id = $3)
            RETURNING id
            """,
            to_vector_literal(at_distance(distance)), answer, study_id,
        )
    return ids


def call(name, arguments):
    """A tool call as the model writes it: arguments are a JSON string, or the raw string given."""
    return (name, arguments if isinstance(arguments, str) else json.dumps(arguments))


def script(stub_llm, turns, tokens=(100, 20)):
    """Make the model play back `turns`. A turn is a list of calls, a string (the final answer) or a function.

    Records what it was sent in stub_llm.calls (the messages) and stub_llm.options (tools and tool_choice).
    """
    stub_llm.options = []
    counter = {"n": 0}

    async def create(model, messages, **kwargs):
        stub_llm.calls.append(list(messages))
        stub_llm.options.append(kwargs)
        turn = turns.pop(0)
        if callable(turn):
            turn = turn(messages)
        calls = None
        if isinstance(turn, list):
            calls = []
            for name, arguments in turn:
                counter["n"] += 1
                calls.append(NS(id=f"call_{counter['n']}", type="function", function=NS(name=name, arguments=arguments)))
        message = NS(content=turn if isinstance(turn, str) else None, tool_calls=calls)
        return NS(choices=[NS(message=message)], usage=NS(prompt_tokens=tokens[0], completion_tokens=tokens[1]))

    stub_llm.create = create
