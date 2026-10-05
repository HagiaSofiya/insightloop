import asyncio

from fastapi.testclient import TestClient

from app import db
from app.main import app

http = TestClient(app)
BRIEF = {"brief": "Why do meal-kit customers cancel after the first month?"}


def test_call_is_recorded_with_prompt_inputs_and_messages(stub_llm, ledger, good_guide):
    stub_llm.outcomes = [good_guide]

    http.post("/briefs", json=BRIEF)

    assert len(ledger) == 1
    row = ledger[0]
    assert row["prompt"].name == "discussion_guide"
    assert row["prompt"].version == "v1"
    assert row["prompt"].inputs == BRIEF
    assert row["model"] == "gpt-4o-mini"
    assert row["messages"][-1] == {"role": "user", "content": BRIEF["brief"]}
    assert row["params"] == {"response_format": "DiscussionGuide"}
    assert row["output"] == good_guide.model_dump_json()
    assert (row["input_tokens"], row["output_tokens"]) == (10, 5)


def test_retry_records_the_retry_messages(stub_llm, ledger, good_guide, validation_error):
    stub_llm.outcomes = [validation_error, good_guide]

    http.post("/briefs", json=BRIEF)

    # The failed attempt gives no usage data, so only the successful retry is recorded.
    assert len(ledger) == 1
    assert "failed validation" in ledger[0]["messages"][-1]["content"]


def test_same_inputs_give_the_same_hash_regardless_of_key_order():
    assert db.inputs_hash({"a": 1, "b": 2}) == db.inputs_hash({"b": 2, "a": 1})
    assert db.inputs_hash({"a": 1}) != db.inputs_hash({"a": 2})


def test_unreachable_database_does_not_raise(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x:x@127.0.0.1:1/x")

    asyncio.run(
        db.record_call(
            prompt=None, model="m", messages=[], params={}, output="o",
            input_tokens=1, output_tokens=1, latency_ms=1.0, cost_usd=None,
        )
    )
