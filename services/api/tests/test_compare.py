import asyncio
from types import SimpleNamespace as NS

import pytest

from app.compare import closed_question_count, format_report, run_all, summarize
from app.prompts.registry import load
from app.schemas import DiscussionGuide

CASES = [{"brief": "first brief"}, {"brief": "second brief"}]


def guide_with(*questions):
    return DiscussionGuide.model_validate(
        {"objectives": ["o"], "questions": [{"text": q, "probes": ["p"]} for q in questions]}
    )


V1_GUIDE = guide_with("Did you like it?", "Were you happy?", "How did it feel?")  # 2 closed
V2_GUIDE = guide_with("How did it feel?", "Why did you stop?", "Describe your week.")  # 0 closed


def script_by_version(stub_llm, v2_outcome=V2_GUIDE):
    """Make the stub answer according to which prompt version it receives."""
    v2_text = load("discussion_guide", "v2")

    async def parse(model, messages, response_format):
        stub_llm.calls.append(messages)
        outcome = v2_outcome if messages[0]["content"] == v2_text else V1_GUIDE
        if isinstance(outcome, Exception):
            raise outcome
        return NS(
            usage=NS(prompt_tokens=10, completion_tokens=5),
            choices=[NS(message=NS(refusal=None, parsed=outcome))],
        )

    stub_llm.parse = parse


def test_closed_question_count_uses_the_first_word():
    assert closed_question_count(V1_GUIDE) == 2
    assert closed_question_count(V2_GUIDE) == 0


def test_every_version_case_and_run_is_called_and_recorded(stub_llm, ledger):
    script_by_version(stub_llm)

    results = asyncio.run(run_all("discussion_guide", ["v1", "v2"], CASES, runs=2, model="m"))

    assert len(stub_llm.calls) == 8  # 2 versions x 2 cases x 2 runs
    assert {k: len(v) for k, v in results.items()} == {(v, i): 2 for v in ("v1", "v2") for i in (0, 1)}
    assert sorted({row["prompt"].version for row in ledger}) == ["v1", "v2"]


def test_report_shows_averages_and_picks_the_version_with_fewer_closed_questions(stub_llm):
    script_by_version(stub_llm)
    results = asyncio.run(run_all("discussion_guide", ["v1", "v2"], CASES, runs=2, model="m"))

    report = format_report("discussion_guide", ["v1", "v2"], CASES, 2, results)

    assert "closed questions per guide: v1 2.0, v2 0.0 -> v2 is better" in report
    assert "failed runs: v1 0, v2 0" in report


def test_failed_runs_are_counted_not_fatal(stub_llm, validation_error):
    script_by_version(stub_llm, v2_outcome=validation_error)  # v2 fails validation twice per run
    results = asyncio.run(run_all("discussion_guide", ["v1", "v2"], CASES, runs=1, model="m"))

    report = format_report("discussion_guide", ["v1", "v2"], CASES, 1, results)

    assert summarize(results[("v2", 0)]) == {"closed": None, "questions": None, "failed": 1}
    assert "failed runs: v1 0, v2 2" in report


def test_a_bug_is_not_hidden_as_a_failed_run(stub_llm):
    async def parse(model, messages, response_format):
        raise TypeError("a programming error")

    stub_llm.parse = parse

    with pytest.raises(TypeError):
        asyncio.run(run_all("discussion_guide", ["v1", "v2"], CASES, runs=1, model="m"))


def test_text_option_prints_the_questions_side_by_side(stub_llm):
    script_by_version(stub_llm)
    results = asyncio.run(run_all("discussion_guide", ["v1", "v2"], CASES[:1], runs=1, model="m"))

    report = format_report("discussion_guide", ["v1", "v2"], CASES[:1], 1, results, show_text=True)

    assert "1. Did you like it?" in report and "| 1. How did it feel?" in report
