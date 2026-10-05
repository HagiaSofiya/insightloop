from fastapi.testclient import TestClient

from app.main import app

http = TestClient(app)
BRIEF = {"brief": "Why do meal-kit customers cancel after the first month?"}


def test_valid_response_returns_the_guide(stub_llm, good_guide):
    stub_llm.outcomes = [good_guide]

    response = http.post("/briefs", json=BRIEF)

    assert response.status_code == 200
    assert response.json() == good_guide.model_dump()
    assert len(stub_llm.calls) == 1


def test_validation_failure_is_retried_once_with_the_error_fed_back(
    stub_llm, good_guide, validation_error
):
    stub_llm.outcomes = [validation_error, good_guide]

    response = http.post("/briefs", json=BRIEF)

    assert response.status_code == 200
    assert len(stub_llm.calls) == 2
    feedback = stub_llm.calls[1][-1]["content"]
    assert "failed validation" in feedback
    assert "questions" in feedback  # the error names the failing field


def test_two_validation_failures_are_reported_cleanly(stub_llm, validation_error):
    stub_llm.outcomes = [validation_error, validation_error]

    response = http.post("/briefs", json=BRIEF)

    assert response.status_code == 502
    assert "invalid guide" in response.json()["detail"]
    assert len(stub_llm.calls) == 2  # one retry, no more


def test_refusal_is_reported_as_422(stub_llm):
    stub_llm.outcomes = ["I can't help with that."]

    response = http.post("/briefs", json=BRIEF)

    assert response.status_code == 422
    assert "refused" in response.json()["detail"]


def test_empty_brief_is_rejected_before_calling_the_model(stub_llm):
    response = http.post("/briefs", json={"brief": ""})

    assert response.status_code == 422
    assert stub_llm.calls == []
