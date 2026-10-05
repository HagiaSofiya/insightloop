from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import main
from app.ask import AskResponse, ask, format_ask, format_evidence
from app.llm.client import LLMInvalidOutput
from app.prompts.registry import load
from app.schemas import Answer, Citation
from app.search import Neighbor
from app.store import insert_simulation
from dbtools import in_rolled_back_transaction, place, script_question, simulation

def script_answer(stub_llm, answer):
    async def parse(model, messages, response_format):
        stub_llm.calls.append(messages)
        return NS(
            usage=NS(prompt_tokens=1, completion_tokens=1),
            choices=[NS(message=NS(refusal=None, parsed=answer))],
        )

    stub_llm.parse = parse


DISTANCES = {"a1": 0.10, "a2": 0.30, "b1": 0.45, "b2": 0.55, "c1": 0.70, "c2": 0.90}


def cited(response_id):
    return Answer(
        answerable=True, answer="People mention it.", citations=[Citation(response_id=response_id, quote="a1")]
    )


# ---- the answer schema ------------------------------------------------------------------------


def test_an_answerable_question_must_cite_something():
    with pytest.raises(ValidationError, match="must cite at least one response"):
        Answer(answerable=True, answer="Trust me.", citations=[])


def test_an_unanswerable_question_needs_no_citations():
    assert Answer(answerable=False, answer="The data covers other topics.", citations=[]).citations == []


def test_a_quote_cannot_be_empty():
    with pytest.raises(ValidationError):
        Citation(response_id=1, quote="")


# ---- retrieval and the threshold ---------------------------------------------------------------


def test_only_responses_within_the_threshold_become_labeled_evidence(stub_llm, ledger):
    script_question(stub_llm)

    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())
        ids = await place(conn, study_id, DISTANCES)
        script_answer(stub_llm, cited(ids["a1"]))

        response = await ask(conn, "Why did they stop?", study_id, embedding_model="m")

        user = stub_llm.calls[0][1]["content"]
        for answer in ("a1", "a2", "b1"):  # distances 0.10, 0.30 and 0.45 are within 0.50
            assert f"[R{ids[answer]}]" in user
        for answer in ("b2", "c1", "c2"):  # 0.55 and beyond are not evidence
            assert f"[R{ids[answer]}]" not in user
        assert user.startswith("Question: Why did they stop?")
        assert stub_llm.calls[0][0]["content"] == load("ask", "v1")
        assert (response.status, response.used) == ("answered", 3)
        assert response.best_distance == pytest.approx(0.10)
        assert response.citations == [Citation(response_id=ids["a1"], quote="a1")]
        assert ledger[0]["prompt"].name == "ask" and ledger[0]["prompt"].inputs["question"] == "Why did they stop?"

    in_rolled_back_transaction(body)


def test_weak_retrieval_refuses_without_calling_the_model(stub_llm):
    script_question(stub_llm)

    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())
        await place(conn, study_id, {"a1": 0.62, "a2": 0.7, "b1": 0.8, "b2": 0.85, "c1": 0.9, "c2": 0.95})

        response = await ask(conn, "What is the capital of France?", study_id, embedding_model="m")

        assert (response.status, response.reason, response.used) == ("not_enough_evidence", "retrieval_too_weak", 0)
        assert response.answer is None and response.citations == []
        assert response.best_distance == pytest.approx(0.62)  # reported, so you can see how close it came
        assert stub_llm.calls == []  # no chat call was made at all

    in_rolled_back_transaction(body)


def test_the_model_can_still_decline_when_the_evidence_does_not_answer(stub_llm):
    script_question(stub_llm)
    script_answer(stub_llm, Answer(answerable=False, answer="They talk about packaging, not price.", citations=[]))

    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())
        await place(conn, study_id, DISTANCES)

        response = await ask(conn, "What about price?", study_id, embedding_model="m")

        assert (response.status, response.reason, response.used) == ("not_enough_evidence", "model_found_no_answer", 3)
        assert response.answer is None and response.citations == []

    in_rolled_back_transaction(body)


def test_a_study_filter_keeps_other_studies_out_of_the_evidence(stub_llm):
    script_question(stub_llm)

    async def body(conn):
        first = await insert_simulation(conn, "first", 0, "m", simulation())
        second = await insert_simulation(conn, "second", 0, "m", simulation())
        first_ids = await place(conn, first, DISTANCES)
        second_ids = await place(conn, second, {**DISTANCES, "a1": 0.01})  # a closer match, in another study
        script_answer(stub_llm, cited(first_ids["a1"]))

        await ask(conn, "q", first, embedding_model="m")
        scoped = stub_llm.calls[-1][1]["content"]
        await ask(conn, "q", None, embedding_model="m")
        unscoped = stub_llm.calls[-1][1]["content"]

        assert f"[R{second_ids['a1']}]" not in scoped and f"[R{first_ids['a1']}]" in scoped
        assert f"[R{second_ids['a1']}]" in unscoped  # without the filter the closer response from the other study wins

    in_rolled_back_transaction(body)


def test_a_stricter_threshold_means_less_evidence(stub_llm):
    script_question(stub_llm)

    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())
        ids = await place(conn, study_id, DISTANCES)
        script_answer(stub_llm, cited(ids["a1"]))

        response = await ask(conn, "q", study_id, embedding_model="m", max_distance=0.2)

        assert response.used == 1

    in_rolled_back_transaction(body)


# ---- citation verification ---------------------------------------------------------------------


def test_cited_quotes_that_appear_in_their_response_are_verified(stub_llm):
    script_question(stub_llm)

    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())
        ids = await place(conn, study_id, DISTANCES)
        script_answer(stub_llm, cited(ids["a1"]))  # the quote "a1" is the whole of that response

        response = await ask(conn, "q", study_id, embedding_model="m")

        assert response.verified is True and response.problems == []

    in_rolled_back_transaction(body)


def test_an_altered_quote_marks_the_answer_unverified_and_says_why(stub_llm):
    script_question(stub_llm)

    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())
        ids = await place(conn, study_id, DISTANCES)
        invented = Answer(answerable=True, answer="They said so.", citations=[Citation(response_id=ids["a1"], quote="something the model made up")])
        script_answer(stub_llm, invented)

        response = await ask(conn, "q", study_id, embedding_model="m")

        assert response.status == "answered"  # the answer is still returned, but flagged
        assert response.verified is False
        assert 'something the model made up' in response.problems[0] and f"not found in R{ids['a1']}" in response.problems[0]

    in_rolled_back_transaction(body)


def test_a_refusal_has_nothing_to_verify(stub_llm):
    script_question(stub_llm)

    async def body(conn):
        study_id = await insert_simulation(conn, "b", 0, "m", simulation())
        await place(conn, study_id, {k: 0.9 for k in DISTANCES})

        response = await ask(conn, "q", study_id, embedding_model="m")

        assert response.status == "not_enough_evidence" and response.verified is None and response.problems == []

    in_rolled_back_transaction(body)


def test_the_command_line_report_says_whether_the_quotes_were_verified():
    base = dict(status="answered", answer="Pricey.", citations=[Citation(response_id=7, quote="Too pricey.")], reason=None, best_distance=0.3, used=2)

    assert "verified: every quote appears in its response" in format_ask(AskResponse(**base, verified=True))
    flagged = format_ask(AskResponse(**base, verified=False, problems=['"Too pricey." [R7]: not found in R7']))
    assert "UNVERIFIED:" in flagged and '- "Too pricey." [R7]: not found in R7' in flagged


# ---- formatting and the HTTP layer -------------------------------------------------------------


def test_evidence_shows_who_said_it_and_what_they_were_asked():
    hits = [
        Neighbor(0.1, 1, "b", 4, "Too pricey.", 34, response_id=7, question="Why did you stop?", gender="woman"),
        Neighbor(0.2, 1, "b", 2, "Fine.", None, response_id=8, question="How was it?", gender=None),
    ]

    assert format_evidence(hits) == (
        "[R7] woman, age 34 | asked: Why did you stop?\nToo pricey.\n\n"
        "[R8] unknown respondent | asked: How was it?\nFine."
    )


def test_the_command_line_report_for_both_outcomes():
    answered = AskResponse(
        status="answered", answer="They find it pricey.", citations=[Citation(response_id=7, quote="Too pricey.")],
        reason=None, best_distance=0.31, used=4,
    )
    refused = AskResponse(
        status="not_enough_evidence", answer=None, citations=[], reason="retrieval_too_weak", best_distance=0.7, used=0,
    )

    assert 'status: answered (4 responses used, best distance 0.310)' in format_ask(answered)
    assert '[R7] "Too pricey."' in format_ask(answered)
    assert "does not answer this question (retrieval_too_weak)" in format_ask(refused)


http = TestClient(main.app)


def fake_answer(monkeypatch, result=None, error=None):
    calls = []

    async def answer_question(question, study_id, k):
        calls.append((question, study_id, k))
        if error:
            raise error
        return result

    monkeypatch.setattr(main, "answer_question", answer_question)
    return calls


def test_the_ask_endpoint_returns_the_answer_and_passes_the_options_through(monkeypatch):
    result = AskResponse(
        status="answered", answer="Pricey.", citations=[Citation(response_id=7, quote="Too pricey.")],
        reason=None, best_distance=0.3, used=2,
    )
    calls = fake_answer(monkeypatch, result)

    response = http.post("/ask", json={"question": "Why stop?", "study_id": 5, "k": 4})

    assert response.status_code == 200
    assert response.json() == result.model_dump()
    assert calls == [("Why stop?", 5, 4)]


def test_the_ask_endpoint_defaults_to_all_studies_and_eight_responses(monkeypatch):
    calls = fake_answer(
        monkeypatch,
        AskResponse(status="not_enough_evidence", answer=None, citations=[], reason="retrieval_too_weak", best_distance=0.8, used=0),
    )

    response = http.post("/ask", json={"question": "Capital of France?"})

    assert response.status_code == 200 and response.json()["status"] == "not_enough_evidence"
    assert calls == [("Capital of France?", None, 8)]


@pytest.mark.parametrize(
    "body",
    [{"question": ""}, {"question": "x" * 501}, {"question": "ok", "k": 0}, {"question": "ok", "k": 21}, {}],
    ids=["empty", "too-long", "k-zero", "k-too-big", "missing"],
)
def test_bad_ask_requests_are_rejected_before_anything_runs(monkeypatch, body):
    calls = fake_answer(monkeypatch, None)

    assert http.post("/ask", json=body).status_code == 422
    assert calls == []


def test_an_invalid_model_answer_is_a_clean_502(monkeypatch):
    fake_answer(monkeypatch, error=LLMInvalidOutput("bad"))

    response = http.post("/ask", json={"question": "q"})

    assert response.status_code == 502 and "invalid answer twice" in response.json()["detail"]
