import re

from app.golden import load_cases, load_golden, relevant_ids, seed_golden
from tests.dbtools import in_rolled_back_transaction

# A plain word that appears in an answer if and only if it mentions the topic. Checks the tags against the text.
MARKERS = {
    (1, "price"): r"expensive|cost|paying|pay |fee|price|paywall|charge|subscription",
    (1, "notifications"): r"reminder|ping|notification|alert|buzz",
    (1, "bugs"): r"crash|froze|glitch|buggy|upload|sync",
    (1, "no_time"): r"did not have|busy|hectic|schedule|keep up",
    (1, "alternative"): r"switched|different plant app|notebook|another app",
}


def test_every_respondent_has_one_answer_and_tag_list_per_question():
    golden = load_golden()
    assert len(golden.respondents) == 30
    assert len({r["name"] for r in golden.respondents}) == 30
    for r in golden.respondents:
        assert len(r["answers"]) == len(r["tags"]) == len(golden.questions)
        assert all(r["answers"]) and all(r["tags"])


def test_tags_agree_with_the_answer_text():
    golden = load_golden()
    for (number, topic), pattern in MARKERS.items():
        for r in golden.respondents:
            mentions = bool(re.search(pattern, r["answers"][number - 1], re.IGNORECASE))
            tagged = topic in r["tags"][number - 1]
            assert mentions == tagged, (topic, r["name"], r["answers"][number - 1])


def test_case_ids_are_unique_and_each_kind_is_present():
    cases = load_cases()
    assert 15 <= len(cases) <= 20
    assert len({c["id"] for c in cases}) == len(cases)
    assert {c["kind"] for c in cases} == {"retrieval", "count", "unanswerable"}


def test_every_expected_count_equals_the_count_from_the_tags():
    golden = load_golden()
    for case in (c for c in load_cases() if c["kind"] == "count"):
        max_age = 30 if case.get("filter") == "under_30" else None
        assert len(golden.tagged(case["question_number"], case["topic"], max_age)) == case["expected_count"], case["id"]


def test_every_retrieval_case_has_relevant_responses():
    golden = load_golden()
    for case in (c for c in load_cases() if c["kind"] == "retrieval"):
        assert golden.tagged(case["question_number"], case["topic"]), case["id"]


def test_unanswerable_questions_are_not_covered_by_any_answer():
    golden = load_golden()
    text = " ".join(a for r in golden.respondents for a in r["answers"]).lower()
    for word in ["dark mode", "ceo", "shipping", "apple watch", "resign"]:
        assert word not in text


def test_seeding_is_idempotent_and_relevant_ids_match_the_tags(stub_llm):
    golden, cases = load_golden(), load_cases()

    async def embed_create(model, input):
        from types import SimpleNamespace as NS
        return NS(data=[NS(index=i, embedding=[0.0] * 1536) for i in range(len(input))], usage=NS(prompt_tokens=1))

    stub_llm.embed_create = embed_create

    async def body(conn):
        first = await seed_golden(conn, golden, "m")
        again = await seed_golden(conn, golden, "m")
        assert first == again
        assert await conn.fetchval("SELECT count(*) FROM responses s JOIN respondents r ON r.id = s.respondent_id WHERE r.study_id = $1", first) == 60
        price = next(c for c in cases if c.get("topic") == "price" and c["kind"] == "retrieval")
        assert len(await relevant_ids(conn, golden, first, price)) == 9 + 8  # 9 price answers plus 8 lower_price ones

    in_rolled_back_transaction(body)
