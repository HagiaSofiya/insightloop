import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from app.ask import AskResponse
from app.evals import (
    EvalReport, Runners, check_gate, closeness, recall_at_k, run_cases, save_baseline, stated_count,
)
from app.golden import load_cases, load_golden, relevant_ids, seed_golden
from app.judge import agreement, read_labels
from app.schemas import Citation
from tests.dbtools import in_rolled_back_transaction


def answered(verified=True, text="People find it too expensive."):
    return AskResponse(status="answered", answer=text, citations=[Citation(response_id=1, quote="x")],
                       reason=None, best_distance=0.3, used=3, verified=verified)


def refused():
    return AskResponse(status="not_enough_evidence", answer=None, citations=[], reason="retrieval_too_weak",
                       best_distance=0.8, used=0)


def runners(top=(1, 2, 3, 4, 5), ask=answered, agent_text="9 respondents stopped.", judge=5):
    async def retrieve(question):
        return list(top)

    async def ask_fn(question):
        return ask() if callable(ask) else ask

    async def agent(question):
        return NS(answer=agent_text)

    async def judge_fn(case, response):
        return judge

    return Runners(retrieve, ask_fn, agent, judge_fn)


CASES = [
    {"id": "r", "kind": "retrieval", "question": "q", "question_number": 1, "topic": "price"},
    {"id": "c", "kind": "count", "question": "how many", "question_number": 1, "topic": "price", "expected_count": 9},
    {"id": "u", "kind": "unanswerable", "question": "dark mode?"},
]


def run(**kwargs):
    return asyncio.run(run_cases(CASES, {"r": {1, 2, 3, 4, 5, 6, 7}}, runners(**kwargs), study_id=7))


def test_recall_divides_by_the_most_that_could_be_found():
    assert recall_at_k([1, 2, 3, 4, 5], {1, 2, 3, 4, 5, 6, 7, 8, 9}) == 1.0  # 9 relevant but only 5 slots
    assert recall_at_k([1, 9, 8, 7, 6], {1, 2}) == 0.5  # 2 relevant, found 1 of them
    assert recall_at_k([], {1, 2}) == 0.0


def test_stated_count_reads_numbers_and_words_but_not_percentages_or_quotes():
    assert stated_count("In study 45, 9 respondents stopped.") == 9
    assert stated_count("Seven respondents said so.") == 7
    assert stated_count('About 30% said so. One said "5 people agree".') is None
    assert stated_count(None) is None


def test_closeness_is_relative_error():
    assert closeness(9, 9) == 1.0
    assert closeness(7, 9) == pytest.approx(1 - 2 / 9)
    assert closeness(30, 9) == 0.0
    assert closeness(None, 9) == 0.0


def test_a_good_run_scores_one_on_every_metric():
    report = run()
    assert report.metrics == {
        "recall_at_5": 1.0, "answer_rate": 1.0, "citation_validity": 1.0, "numeric_closeness": 1.0,
        "numeric_exact": 1.0, "refusal_rate": 0.0, "judge_quality": 1.0,
    }  # refusal 0.0 because this fake answers even the unanswerable question


def test_each_metric_moves_for_its_own_failure():
    assert run(top=(9, 10, 11, 12, 13)).metrics["recall_at_5"] == 0.0
    assert run(ask=lambda: answered(verified=False)).metrics["citation_validity"] == 0.0
    assert run(ask=refused).metrics["refusal_rate"] == 1.0
    assert run(ask=refused).metrics["answer_rate"] == 0.0
    assert run(ask=refused).metrics["citation_validity"] is None  # nothing was answered, so nothing to validate
    assert run(agent_text="5 respondents stopped.").metrics["numeric_exact"] == 0.0
    assert run(judge=1).metrics["judge_quality"] == 0.0


def test_judge_is_optional():
    r = runners()
    r.judge = None
    report = asyncio.run(run_cases(CASES, {"r": {1}}, r, 7))
    assert report.metrics["judge_quality"] is None


def test_unknown_case_kind_is_an_error():
    with pytest.raises(ValueError, match="unknown kind"):
        asyncio.run(run_cases([{"id": "x", "kind": "nope", "question": "q"}], {}, runners(), 1))


# ---- the gate ----------------------------------------------------------------------------------

BASELINE = {"metrics": {"recall_at_5": 0.8, "citation_validity": 1.0, "judge_quality": None}}


def report_with(**metrics):
    from app.evals import CaseResult
    scores = {"recall_at_5": None, "citation_validity": None, "judge_quality": None, **metrics}
    return EvalReport([CaseResult("c", "retrieval", "q", scores)])


def test_gate_passes_when_nothing_dropped_more_than_allowed():
    assert check_gate(report_with(recall_at_5=0.75, citation_validity=1.0), BASELINE) == []
    assert check_gate(report_with(recall_at_5=0.70, citation_validity=1.0), BASELINE) == []  # exactly the allowed drop


def test_gate_fails_on_a_drop_larger_than_allowed():
    failures = check_gate(report_with(recall_at_5=0.5, citation_validity=1.0), BASELINE)
    assert len(failures) == 1 and failures[0].startswith("recall_at_5")


def test_gate_fails_when_a_baseline_metric_is_no_longer_measured():
    failures = check_gate(report_with(recall_at_5=0.8), BASELINE)
    assert failures == ["citation_validity: not measured (baseline 1.00)"]


def test_improvements_never_fail_the_gate():
    assert check_gate(report_with(recall_at_5=1.0, citation_validity=1.0), BASELINE) == []


def test_save_baseline_records_metrics_and_versions(tmp_path):
    path = tmp_path / "baseline.json"
    save_baseline(report_with(recall_at_5=0.8), path)
    saved = json.loads(path.read_text())
    assert saved["metrics"]["recall_at_5"] == 0.8
    assert {"ask", "agent", "judge", "answer_model"} <= set(saved["versions"])


# ---- judge agreement ---------------------------------------------------------------------------


def test_agreement_counts_exact_near_and_pass_fail_matches():
    result = agreement([(5, 5), (4, 3), (2, 5), (3, 3)])
    assert result == {"n": 4, "exact": 0.5, "within_1": 0.75, "pass_fail": 0.5}


def test_agreement_needs_labels():
    with pytest.raises(ValueError):
        agreement([])


def test_unlabeled_rows_are_skipped(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([{"judge_score": 4, "human_score": 5}, {"judge_score": 3, "human_score": None}]))
    assert read_labels(path) == [(4, 5)]


# ---- relevant ids with also_relevant -------------------------------------------------------------


def test_also_relevant_adds_the_paired_topic(stub_llm):
    golden = load_golden()
    price = next(c for c in load_cases() if c["kind"] == "retrieval" and c["topic"] == "price")
    assert price["also_relevant"] == [{"question_number": 2, "topic": "lower_price"}]

    async def embed_create(model, input):
        return NS(data=[NS(index=i, embedding=[0.0] * 1536) for i in range(len(input))], usage=NS(prompt_tokens=1))

    stub_llm.embed_create = embed_create

    async def body(conn):
        study_id = await seed_golden(conn, golden, "m")
        assert len(await relevant_ids(conn, golden, study_id, price)) == 9 + 8

    in_rolled_back_transaction(body)
