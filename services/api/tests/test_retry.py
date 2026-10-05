import asyncio
from types import SimpleNamespace as NS

import httpx2 as httpx
import openai
import pytest

from app.llm import retry
from app.llm.client import complete, complete_structured
from app.respondents import interview_all
from app.schemas import DiscussionGuide, Persona

REQUEST = httpx.Request("POST", "http://test")


def status_error(cls, status, body=None):
    return cls("error", response=httpx.Response(status, request=REQUEST), body=body)


def rate_limit():
    return status_error(openai.RateLimitError, 429, {"type": "requests", "code": "rate_limit_exceeded"})


def out_of_credits():
    body = {"type": "insufficient_quota", "code": "credit_balance_exhausted"}
    return status_error(openai.RateLimitError, 429, body)


def server_error():
    return status_error(openai.InternalServerError, 500)


def script_create(stub_llm, outcomes):
    """Make the plain-text call raise or return each queued outcome in turn."""

    async def create(model, messages):
        stub_llm.calls.append(messages)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return NS(
            usage=NS(prompt_tokens=1, completion_tokens=1),
            choices=[NS(message=NS(content=outcome))],
        )

    stub_llm.create = create


def ask():
    return asyncio.run(complete([{"role": "user", "content": "hi"}], "m"))


@pytest.mark.parametrize(
    "transient",
    [
        rate_limit(),
        server_error(),
        openai.APIConnectionError(request=REQUEST),
        openai.APITimeoutError(request=REQUEST),
    ],
    ids=["429", "5xx", "connection", "timeout"],
)
def test_transient_errors_are_retried_until_they_succeed(stub_llm, ledger, sleeps, transient):
    script_create(stub_llm, [transient, transient, "finally"])

    result = ask()

    assert result.text == "finally"
    assert len(stub_llm.calls) == 3
    assert len(sleeps) == 2
    assert len(ledger) == 1  # only the successful attempt is recorded


@pytest.mark.parametrize(
    "permanent",
    [
        out_of_credits(),
        status_error(openai.BadRequestError, 400),
        status_error(openai.AuthenticationError, 401),
    ],
    ids=["out-of-credits", "400", "401"],
)
def test_errors_that_waiting_cannot_fix_are_not_retried(stub_llm, sleeps, permanent):
    script_create(stub_llm, [permanent])

    with pytest.raises(type(permanent)):
        ask()

    assert len(stub_llm.calls) == 1
    assert sleeps == []


def test_gives_up_after_the_attempt_budget(stub_llm, sleeps):
    script_create(stub_llm, [rate_limit() for _ in range(retry.MAX_ATTEMPTS)])

    with pytest.raises(openai.RateLimitError):
        ask()

    assert len(stub_llm.calls) == retry.MAX_ATTEMPTS
    assert len(sleeps) == retry.MAX_ATTEMPTS - 1  # no wait after the last failure


def test_backoff_ceiling_doubles_and_is_capped(stub_llm, sleeps, monkeypatch):
    monkeypatch.setattr(retry.random, "uniform", lambda low, high: high)  # always the ceiling
    monkeypatch.setattr(retry, "MAX_ATTEMPTS", 7)
    script_create(stub_llm, [server_error() for _ in range(6)] + ["ok"])

    ask()

    assert sleeps == [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]  # doubles, then stops at MAX_DELAY


def test_jitter_picks_a_random_wait_below_the_ceiling(stub_llm, sleeps):
    for _ in range(40):
        script_create(stub_llm, [server_error(), server_error(), server_error(), "ok"])
        ask()

    ceilings = [0.5, 1.0, 2.0] * 40
    assert all(0 <= wait <= ceiling for wait, ceiling in zip(sleeps, ceilings))
    assert len(set(sleeps)) > 20  # the waits differ, so clients would not retry in lockstep


def test_structured_calls_are_retried_too(stub_llm, good_guide, sleeps):
    outcomes = [rate_limit(), good_guide]

    async def parse(model, messages, response_format):
        stub_llm.calls.append(messages)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return NS(
            usage=NS(prompt_tokens=1, completion_tokens=1),
            choices=[NS(message=NS(refusal=None, parsed=outcome))],
        )

    stub_llm.parse = parse

    result = asyncio.run(complete_structured([{"role": "user", "content": "x"}], "m", DiscussionGuide))

    assert result == good_guide
    assert len(stub_llm.calls) == 2 and len(sleeps) == 1


def test_a_failure_in_the_middle_of_the_interviews_is_survived(stub_llm, sleeps):
    """The milestone check in miniature: one injected 429 and every answer still arrives."""
    state = {"calls": 0}

    async def create(model, messages):
        stub_llm.calls.append(messages)
        state["calls"] += 1
        if state["calls"] == 5:
            raise rate_limit()
        return NS(
            usage=NS(prompt_tokens=1, completion_tokens=1),
            choices=[NS(message=NS(content=f"answer to: {messages[-1]['content']}"))],
        )

    stub_llm.create = create
    personas = [
        Persona(
            name=f"P{i}", occupation="o", backstory="b", speaking_style="s",
            age=30, gender="woman", city="c", income="i", household="h",
        )
        for i in range(4)
    ]
    questions = ["q1", "q2", "q3"]

    answers = asyncio.run(interview_all(personas, questions, "m", concurrency=4))

    assert answers == [[f"answer to: {q}" for q in questions]] * 4
    assert len(stub_llm.calls) == 4 * 3 + 1  # one extra call for the retry
    assert len(sleeps) == 1
