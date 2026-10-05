import asyncio
from types import SimpleNamespace as NS

import pytest

from app import multi_agent
from app.llm.client import complete_streaming
from app.multi_agent import research
from app.run_events import events_after
from app.schemas import Critique, Plan
from app.workflow import TokenBatcher, advance
from dbtools import in_rolled_back_transaction, script
from test_workflow import Fake, new_run
from test_multi_agent import same_connection


def chunk(text=None, usage=None):
    choices = [NS(delta=NS(content=text))] if usage is None else []
    return NS(choices=choices, usage=usage)


def stream_of(stub_llm, *pieces, usage=NS(prompt_tokens=11, completion_tokens=4), closed=None):
    class Stream:
        def __init__(self):
            self.items = [chunk(p) for p in pieces] + ([chunk(usage=usage)] if usage else [])

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.items:
                raise StopAsyncIteration
            return self.items.pop(0)

        async def close(self):
            if closed is not None:
                closed.append(True)

    async def create(model, messages, **kwargs):
        stub_llm.options = kwargs
        return Stream()

    stub_llm.create = create


def test_streaming_passes_each_piece_in_order_and_returns_the_whole_reply(stub_llm, ledger):
    stream_of(stub_llm, "Hel", "lo ", "world")
    seen = []

    async def on_text(text):
        seen.append(text)

    result = asyncio.run(complete_streaming([{"role": "user", "content": "hi"}], "gpt-4o-mini", on_text))
    assert seen == ["Hel", "lo ", "world"] and result.text == "Hello world"
    assert stub_llm.options["stream"] is True and stub_llm.options["stream_options"] == {"include_usage": True}
    assert (result.input_tokens, result.output_tokens) == (11, 4)
    assert ledger[0]["output"] == "Hello world" and ledger[0]["input_tokens"] == 11  # logged like every other call


def test_streaming_stops_reading_if_the_callback_fails(stub_llm, ledger):
    closed = []
    stream_of(stub_llm, "a", "b", closed=closed)

    async def on_text(text):
        raise ConnectionError("client went away")

    with pytest.raises(ConnectionError):
        asyncio.run(complete_streaming([], "m", on_text))
    assert closed == [True] and ledger == []


def test_a_stream_without_token_counts_is_an_error_not_a_silent_free_call(stub_llm, ledger):
    stream_of(stub_llm, "cut off", usage=None)

    async def on_text(text):
        pass

    with pytest.raises(RuntimeError, match="without token counts"):
        asyncio.run(complete_streaming([], "m", on_text))
    assert ledger == []


def test_the_synthesizer_streams_and_the_critic_triggered_redraft_sends_a_reset(stub_llm):
    events = []

    async def on_event(kind, text):
        events.append((kind, text))

    stub_llm.outcomes = [Plan(sub_questions=["p1?"]), Critique(ok=False, gaps=["gap?"])]
    script(stub_llm, ["finding", "draft", "gap finding", "final"])  # analyst, synthesizer, analyst, synthesizer

    async def body(conn):
        multi_agent_streams = []

        async def streaming_create(model, messages, **kwargs):
            if not kwargs.get("stream"):
                return await scripted(model, messages, **kwargs)
            text = ("draft" if not multi_agent_streams else "final")
            multi_agent_streams.append(text)
            return await stream_object(text)

        scripted = stub_llm.create
        stub_llm.create = streaming_create

        async def stream_object(text):
            class S:
                def __init__(self):
                    self.items = [chunk(text[:2]), chunk(text[2:]), chunk(usage=NS(prompt_tokens=1, completion_tokens=1))]

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if not self.items:
                        raise StopAsyncIteration
                    return self.items.pop(0)

            return S()

        result = await research(conn, "q?", connection=same_connection(conn), concurrency=1, on_event=on_event)
        assert result.answer == "final"

    in_rolled_back_transaction(body)
    assert events == [("delta", "dr"), ("delta", "aft"), ("reset", ""), ("delta", "fi"), ("delta", "nal")]


def test_the_batcher_groups_small_pieces_by_size_or_time():
    sent, now = [], {"t": 0.0}

    async def flush(text):
        sent.append(text)

    async def run():
        batcher = TokenBatcher(flush, min_chars=10, max_seconds=1.0, clock=lambda: now["t"])
        for piece in ["ab", "cd", "ef", "gh", "ij", "kl"]:  # the fifth piece reaches 10 characters
            await batcher.add(piece)
        now["t"] = 2.0
        await batcher.add("m")  # more than a second since the last flush
        await batcher.add("n")
        await batcher.finish()  # the end flushes what is left

    asyncio.run(run())
    assert sent == ["abcdefghij", "klm", "n"]


def test_the_report_can_be_rebuilt_from_the_streamed_events_and_matches_the_saved_one():
    async def body(conn):
        run_id, fake = await new_run(conn, count=3), Fake()
        await advance(conn, run_id, fake.activities, concurrency=1)
        import json

        events = await events_after(conn, run_id, 0, 1000)
        text = ""
        for event in events:
            if event["type"] == "report.reset":
                text = ""
            elif event["type"] == "report.delta":
                text += json.loads(event["data"])["text"]
        assert text == await conn.fetchval("SELECT report FROM workflow_runs WHERE id = $1", run_id) == "the report"

    in_rolled_back_transaction(body)


def test_a_retried_synthesis_starts_with_a_reset_so_clients_drop_the_partial_report():
    async def body(conn):
        import json

        run_id, fake = await new_run(conn, count=3), Fake()
        original = fake.synthesize
        state = {"first": True}

        async def flaky(conn, study_id, on_event):
            if state["first"]:
                state["first"] = False
                await on_event("delta", "half a rep")
                raise RuntimeError("stream broke")
            return await original(conn, study_id, on_event)

        acts = fake.activities
        acts.synthesize = flaky
        with pytest.raises(RuntimeError):
            await advance(conn, run_id, acts, concurrency=1)
        await advance(conn, run_id, acts, concurrency=1)
        types = [e["type"] for e in await events_after(conn, run_id, 0, 1000) if e["type"].startswith("report.")]
        # the failed attempt buffered its half sentence and died before flushing; the retry resets, then streams the report
        assert types == ["report.reset", "report.reset", "report.delta"]

    in_rolled_back_transaction(body)
