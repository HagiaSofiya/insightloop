import asyncio
import json

import pytest

from app import cli


def stored_row(**overrides):
    row = {
        "id": 7,
        "prompt_name": "discussion_guide",
        "prompt_version": "v1",
        "inputs": json.dumps({"brief": "why do people churn?"}),
        "model": "gpt-4o-mini",
        "messages": json.dumps(
            [{"role": "system", "content": "S"}, {"role": "user", "content": "why do people churn?"}]
        ),
        "params": json.dumps({"response_format": "DiscussionGuide"}),
        "output": "the original output",
    }
    return {**row, **overrides}


@pytest.fixture
def stored(monkeypatch):
    def put(row):
        async def fake_fetch_call(call_id):
            return row if call_id == row["id"] else None

        monkeypatch.setattr(cli, "fetch_call", fake_fetch_call)

    return put


def test_replay_sends_the_stored_messages_and_model_unchanged(stored, stub_llm, ledger, good_guide):
    row = stored_row()
    stored(row)
    stub_llm.outcomes = [good_guide]

    returned_row, new_output = asyncio.run(cli.replay(7))

    assert stub_llm.calls == [json.loads(row["messages"])]
    assert new_output == good_guide.model_dump_json()
    assert returned_row["output"] == "the original output"


def test_replay_is_recorded_with_the_same_prompt_and_inputs(stored, stub_llm, ledger, good_guide):
    stored(stored_row())
    stub_llm.outcomes = [good_guide]

    asyncio.run(cli.replay(7))

    assert len(ledger) == 1
    assert ledger[0]["prompt"].name == "discussion_guide"
    assert ledger[0]["prompt"].version == "v1"
    assert ledger[0]["prompt"].inputs == {"brief": "why do people churn?"}


def test_plain_text_call_is_replayed_without_a_schema(stored, stub_llm):
    stored(stored_row(params="{}", prompt_name=None, prompt_version=None, inputs=None))
    stub_llm.outcomes = ["a fresh answer"]

    _, new_output = asyncio.run(cli.replay(7))

    assert new_output == "a fresh answer"


def test_unknown_call_id_raises(stored):
    stored(stored_row())

    with pytest.raises(cli.CallNotFound):
        asyncio.run(cli.replay(999))
