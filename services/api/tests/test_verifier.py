import json

import pytest

from app.schemas import Citation
from app.store import insert_simulation
from app.verifier import (
    NumberPool, build_pool, extract_count_claims, extract_quotes, format_verification, normalize,
    quote_in_text, verify_answer, verify_citations, Verification, Problem,
)
from dbtools import in_rolled_back_transaction, simulation

TEXTS = {
    "a1": "I love the colors, but the price is too high for what you get. My kids would not eat it.",
    "b1": "Honestly it didn’t fit my routine at all.",  # a curly apostrophe, like many real responses
    "c1": "Delivery was late. Packaging was fine. I stopped after a month.",
}


async def seed(conn):
    """A study whose answers have known wording. Returns {'a1': response id, ...}."""
    study_id = await insert_simulation(conn, "b", 0, "m", simulation())
    ids = {}
    for original, text in TEXTS.items():
        ids[original] = await conn.fetchval(
            "UPDATE responses SET answer = $1 WHERE answer = $2 AND respondent_id IN "
            "(SELECT id FROM respondents WHERE study_id = $3) RETURNING id",
            text, original, study_id,
        )
    return ids


# what count_by_theme would have returned for a question with 30 answers
COUNTS = {
    "study_id": 7, "question_number": 3, "group_by": "age_band", "responses": 30,
    "themes": [
        {"theme": "Price", "size": 9, "counts": {"18-29": 6, "30-44": 3}},
        {"theme": "Fit", "size": 6, "counts": {"18-29": 2, "45-59": 4}},
    ],
    "totals": {"18-29": 8, "30-44": 3, "45-59": 4},
}
SEARCH = {"count": 2, "results": [{"response_id": 11, "study_id": 7, "question_number": 3, "age": 25, "gender": "woman", "match": "strong", "answer": "The price is too high."}]}


def results(*payloads):
    return [json.dumps(p) for p in payloads]


# ---- quotes: reading them and comparing them ---------------------------------------------------


def test_normalize_only_changes_spacing_and_typography():
    assert normalize("  it’s   “fine”\n— really  ") == 'it\'s "fine" - really'


@pytest.mark.parametrize(
    "quote, text, expected",
    [
        ("the price is too high", "Well, the price is too high for me.", True),
        ("the  price\nis too high", "the price is too high", True),  # spacing differs
        ("it's fine", "it’s fine", True),  # straight vs curly apostrophe
        ("the price is too high.", "the price is too high for me", True),  # a period that is not in the text
        ("price is too high", "Price is too high", True),  # the first letter's case may differ: quotes get lowercased to fit a sentence
        ("It just felt more stressful", "it just felt more stressful than helpful", True),  # and the other way round
        ("the Price is too high", "the price is too high", False),  # any other case change is an alteration
        ("THE PRICE IS TOO HIGH", "the price is too high", False),
        ("the cost is too high", "the price is too high", False),  # a changed word
        ("too high the price", "the price is too high", False),
        ("I want something that absorbs impact...given my knee pain", "I want something that absorbs impact, as I have knees. Given my knee pain I stop.", True),  # skipped text, and a first letter in a different case
        ("I want something that absorbs impact...given my back pain", "I want something that absorbs impact, as I have knees. Given my knee pain I stop.", False),  # the second piece is not there
        ("I want something...knee pain", "I want something that absorbs impact given my knee pain", True),  # skipped text, in order
        ("knee pain...I want something", "I want something that absorbs impact given my knee pain", False),  # out of order
        ("", "anything", False),
        ("...", "anything", False),
    ],
)
def test_a_quote_must_appear_word_for_word(quote, text, expected):
    assert quote_in_text(quote, text) is expected


def test_quotes_are_read_with_their_citations():
    answer = 'One said "the price is too high" [R12] and "it didn’t fit" [R3, R9]. Also “a curly one here” [R5] and "no citation on this one".'

    found = extract_quotes(answer)

    assert [(q.text, q.response_ids) for q in found] == [
        ("the price is too high", [12]), ("it didn’t fit", [3, 9]), ("a curly one here", [5]), ("no citation on this one", []),
    ]
    assert answer[found[0].start : found[0].end] == '"the price is too high"'


def test_a_two_character_quote_is_not_treated_as_a_quote():
    assert extract_quotes('Called "ok" by one [R1].') == []


# ---- numbers: reading the claims and the pool of allowed values ---------------------------------


def claims(answer):
    return sorted((text, value, pct) for text, value, pct in extract_count_claims(answer, extract_quotes(answer)))


def test_percentages_and_counts_before_counting_words_are_claims():
    assert claims("About 30% of respondents and 12 people mention cost; five responses disagree.") == [
        ("12 people", 12.0, False), ("30%", 30.0, True), ("five responses", 5.0, False),
    ]
    assert claims("It was 45 percent.") == [("45 percent", 45.0, True)]
    assert claims("8 of 30 respondents agreed.") == [("30 respondents", 30.0, False), ("8 of 30", 8.0, False)]


def test_numbers_that_are_identifiers_ages_or_inside_quotes_are_not_claims():
    answer = 'In study 45 question 3 respondents aged 20 to 29 (younger means under 30) said "all 12 people hated it" [R377]. See Q4 and [R12, R15].'

    assert claims(answer) == []


def test_one_respondent_is_phrasing_not_a_statistic():
    assert claims("One respondent said it, and 1 respondent agreed.") == []


def test_the_pool_holds_numbers_from_tool_results_and_percentages_worked_out_from_them():
    pool = build_pool([COUNTS, SEARCH])

    assert {30, 9, 6, 3, 2, 4, 8} <= pool.counts
    assert 25 not in pool.counts and 7 not in pool.counts  # an age and a study id are not counts
    def has(percent):
        return any(abs(p - percent) < 0.05 for p in pool.percents)

    assert has(30.0) and has(20.0)  # 9 of 30 and 6 of 30
    assert has(75.0)  # 6 of the 8 respondents aged 18-29 chose Price
    assert has(66.7 - 0.033)  # 6 of Price's 9 are in the youngest band


# ---- verifying an answer ------------------------------------------------------------------------


GOOD = (
    'Younger respondents (under 30) find it too costly: "the price is too high for what you get" [{a1}]. '
    '9 respondents chose the Price theme, which is 30% of the 30 answers.'
)
TAMPERED = (
    'Younger respondents (under 30) find it too costly: "the price is far too high for what you get" [{a1}]. '
    '14 respondents chose the Price theme, which is 45% of the 30 answers.'
)


def test_a_good_answer_passes_and_a_tampered_one_is_caught():
    """The milestone's check: the same answer with a changed quote and changed numbers fails."""

    async def body(conn):
        ids = await seed(conn)
        tools = results(SEARCH, COUNTS)

        good = await verify_answer(conn, GOOD.format(a1=f"R{ids['a1']}"), tools)
        tampered = await verify_answer(conn, TAMPERED.format(a1=f"R{ids['a1']}"), tools)

        assert good.verified and good.problems == []
        assert (good.quotes_checked, good.numbers_checked) == (1, 3)  # one quote, and the 9, the 30% and the 30
        assert not tampered.verified
        kinds = sorted((p.kind, p.claim.split(" [")[0]) for p in tampered.problems)
        assert kinds == [("number", "14 respondents"), ("number", "45%"), ("quote", '"the price is far too high for what you get"')]

    in_rolled_back_transaction(body)


def test_a_quote_credited_to_the_wrong_response_says_where_the_words_really_are():
    async def body(conn):
        ids = await seed(conn)

        verification = await verify_answer(conn, f'They said "I stopped after a month" [R{ids["a1"]}].', [])

        (problem,) = verification.problems
        assert problem.kind == "quote"
        assert f"not found in R{ids['a1']}" in problem.detail and f"R{ids['c1']} instead" in problem.detail

    in_rolled_back_transaction(body)


def test_a_quote_nobody_said_and_a_response_that_does_not_exist_are_both_reported():
    async def body(conn):
        ids = await seed(conn)

        invented = await verify_answer(conn, f'"this was never said by anyone" [R{ids["a1"]}]', [])
        ghost = await verify_answer(conn, '"anything at all here" [R999999999]', [])

        assert "these words appear in no response" in invented.problems[0].detail
        assert "R999999999 does not exist" in ghost.problems[0].detail

    in_rolled_back_transaction(body)


def test_curly_apostrophes_and_skipped_text_in_a_real_quote_are_accepted():
    async def body(conn):
        ids = await seed(conn)

        verification = await verify_answer(
            conn, f'"it didn\'t fit my routine" [R{ids["b1"]}] and "I love the colors...My kids would not eat it" [R{ids["a1"]}]', [],
        )

        assert verification.verified and verification.quotes_checked == 2

    in_rolled_back_transaction(body)


def test_an_uncited_long_quote_must_come_from_something_the_agent_retrieved():
    async def body(conn):
        await seed(conn)
        tools = results({"theme": {"label": "Price worries", "description": "Respondents feel the cost is not worth it."}})

        seen = await verify_answer(conn, 'The theme says "the cost is not worth it".', tools)
        unseen = await verify_answer(conn, 'The theme says "the cost is outrageous and unfair".', tools)
        scare = await verify_answer(conn, 'Defining "younger" as under 30.', tools)  # a short quoted term is not a claim

        assert seen.verified and unseen.problems[0].detail.startswith("quoted without a citation")
        assert scare.verified and scare.quotes_checked == 0

    in_rolled_back_transaction(body)


def test_numbers_must_match_the_tool_results_and_percentages_may_be_off_by_one():
    async def body(conn):
        tools = results(COUNTS)

        async def check(answer):
            return await verify_answer(conn, answer, tools)

        assert (await check("9 respondents (30% of the sample).")).verified
        assert (await check("About 31% of respondents.")).verified  # within a point of 30.0%
        assert (await check("About 33% of respondents.")).verified  # 3 of Price's 9 is 33.3%, so this is derivable
        assert not (await check("About 45% of respondents.")).verified  # nothing in the results gives 45
        assert (await check("6 of the 8 younger respondents chose it.")).verified  # 6 and 8 are both in the results
        assert not (await check("11 respondents chose it.")).verified
        assert (await check("That is 75% of respondents under 30.")).verified  # 6 of 8, worked out from the counts

    in_rolled_back_transaction(body)


def test_tool_errors_do_not_count_as_evidence():
    async def body(conn):
        tools = results({"error": "No themes exist. The numbers 12 and 40 are here."}, COUNTS)

        verification = await verify_answer(conn, "12 respondents said so.", tools)

        assert not verification.verified

    in_rolled_back_transaction(body)


def test_an_answer_with_no_quotes_or_counts_has_nothing_to_check():
    async def body(conn):
        verification = await verify_answer(conn, "The data does not cover this topic.", [])

        assert verification.verified and (verification.quotes_checked, verification.numbers_checked) == (0, 0)

    in_rolled_back_transaction(body)


def test_ask_citations_are_checked_the_same_way():
    async def body(conn):
        ids = await seed(conn)

        good = await verify_citations(conn, [Citation(response_id=ids["a1"], quote="the price is too high")])
        bad = await verify_citations(conn, [Citation(response_id=ids["a1"], quote="the price is outrageous"), Citation(response_id=ids["c1"], quote="Packaging was fine")])

        assert good.verified and good.quotes_checked == 1
        assert [p.claim.split(" [")[0] for p in bad.problems] == ['"the price is outrageous"']

    in_rolled_back_transaction(body)


def test_the_verification_report_lists_what_is_wrong():
    ok = Verification(True, [], 2, 1)
    bad = Verification(False, [Problem("quote", '"x y z w"', "not found in R1"), Problem("number", "45%", "no match")], 3, 2)

    assert format_verification(ok) == "verified (2 quotes and 1 numbers checked)"
    assert format_verification(bad) == (
        'UNVERIFIED (3 quotes and 2 numbers checked):\n  - quote: "x y z w": not found in R1\n  - number: 45%: no match'
    )
