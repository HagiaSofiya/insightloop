from types import SimpleNamespace as NS

import pytest
from pydantic import ValidationError

from app.llm import client, retry
from app.schemas import DiscussionGuide


class StubLLM:
    """Stands in for the OpenAI client. Tests queue what each call should return.

    A queued Exception is raised, a queued string is a refusal and a queued
    DiscussionGuide is a successful parse. An empty queue fails the test, so no
    test can reach the real API by accident.
    """

    def __init__(self):
        self.outcomes = []
        self.calls = []  # the messages sent on each call

    async def parse(self, model, messages, response_format):
        self.calls.append(messages)
        if not self.outcomes:
            raise AssertionError("unexpected LLM call: nothing queued on the stub")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        refusal = outcome if isinstance(outcome, str) else None
        parsed = None if refusal else outcome
        return NS(
            usage=NS(prompt_tokens=10, completion_tokens=5),
            choices=[NS(message=NS(refusal=refusal, parsed=parsed))],
        )


    async def embed_create(self, model, input):
        """Embeddings path. Tests replace this with a function that returns vectors."""
        raise AssertionError("unexpected embeddings call: the test did not script one")

    async def create(self, model, messages, **kwargs):
        """Plain-text path: a queued string is the reply text."""
        self.calls.append(messages)
        if not self.outcomes:
            raise AssertionError("unexpected LLM call: nothing queued on the stub")
        return NS(
            usage=NS(prompt_tokens=10, completion_tokens=5),
            choices=[NS(message=NS(content=self.outcomes.pop(0)))],
        )


@pytest.fixture(autouse=True)
def stub_llm(monkeypatch):
    stub = StubLLM()
    monkeypatch.setattr(
        client,
        "_client",
        NS(
            chat=NS(completions=stub),
            # looked up at call time, so a test can replace stub.embed_create
            embeddings=NS(create=lambda **kwargs: stub.embed_create(**kwargs)),
        ),
    )
    return stub


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Retry waits are recorded instead of slept, so tests stay fast and can check the delays."""
    delays = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(retry, "sleep", fake_sleep)
    return delays


@pytest.fixture(autouse=True)
def ledger(monkeypatch):
    """Replaces the llm_calls write with a list, so tests never touch Postgres."""
    rows = []

    async def fake_record_call(**row):
        rows.append(row)

    monkeypatch.setattr(client, "record_call", fake_record_call)
    return rows


@pytest.fixture
def good_guide():
    return DiscussionGuide.model_validate(
        {
            "objectives": ["Learn why people cancel"],
            "questions": [{"text": "Why did you cancel?", "probes": ["What changed?"]}],
        }
    )


@pytest.fixture
def validation_error():
    """A real Pydantic error, like the one a bad model response would produce."""
    try:
        DiscussionGuide.model_validate({"objectives": ["x"], "questions": []})
    except ValidationError as error:
        return error
