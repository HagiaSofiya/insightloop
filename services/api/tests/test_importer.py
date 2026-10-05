from pathlib import Path

import pytest

from app.importer import format_import, read_csv

SAMPLE = Path(__file__).parents[3] / "evals" / "sample_responses.csv"


def write(tmp_path, text, encoding="utf-8"):
    path = tmp_path / "responses.csv"
    path.write_text(text, encoding=encoding)
    return str(path)


def test_columns_split_into_attributes_ignored_noise_and_questions(tmp_path):
    path = write(
        tmp_path,
        "Response ID,Timestamp,Age,Gender,Why did you try it?,Why did you stop?\n"
        "1,2026-09-01,34,woman,A coworker said so,Too pricey\n"
        "2,2026-09-02,58,man,My wife wanted to,Small portions\n",
    )

    data = read_csv(path)

    assert data.questions == ["Why did you try it?", "Why did you stop?"]  # id and timestamp are not questions
    assert data.profiles == [{"age": 34, "gender": "woman"}, {"age": 58, "gender": "man"}]
    assert data.answers == [["A coworker said so", "Too pricey"], ["My wife wanted to", "Small portions"]]


def test_attribute_names_ignore_case_and_spaces_and_all_are_optional(tmp_path):
    with_attributes = read_csv(write(tmp_path, " CITY ,Q\nAustin,Because\n"))
    without = read_csv(write(tmp_path, "Q\nBecause\n"))

    assert with_attributes.profiles == [{"city": "Austin"}]
    assert without.profiles == [{}]
    assert without.questions == ["Q"]


def test_blank_answers_and_blank_ages_become_none(tmp_path):
    data = read_csv(write(tmp_path, "Age,Q1,Q2\n,only one answer,\n30,both,here\n"))

    assert data.profiles == [{"age": None}, {"age": 30}]
    assert data.answers == [["only one answer", None], ["both", "here"]]
    assert data.blank_answers == 1


def test_blank_lines_and_short_rows_are_handled(tmp_path):
    data = read_csv(write(tmp_path, "Age,Q1,Q2\n30,a,b\n\n,,\n40,c\n"))

    assert data.answers == [["a", "b"], ["c", None]]  # the blank lines vanish and the short row is padded


def test_a_file_saved_by_excel_with_a_bom_is_read(tmp_path):
    data = read_csv(write(tmp_path, "Age,Q1\n30,hello\n", encoding="utf-8-sig"))

    assert data.questions == ["Q1"]  # without utf-8-sig the first header would be "﻿Age"
    assert data.profiles == [{"age": 30}]


def test_answers_keep_commas_quotes_and_newlines(tmp_path):
    data = read_csv(write(tmp_path, 'Q\n"Well, I said ""no"" twice\nthen left"\n'))

    assert data.answers == [['Well, I said "no" twice\nthen left']]


def test_a_bad_age_names_the_line(tmp_path):
    path = write(tmp_path, "Age,Q\n30,fine\nthirty,oops\n")

    with pytest.raises(ValueError, match="Line 3: age 'thirty'"):
        read_csv(path)


@pytest.mark.parametrize(
    "text, message",
    [
        ("Age,Gender\n30,woman\n", "No question columns"),
        ("Q1,Q2\n", "no respondent rows"),
        ("", "No question columns"),
    ],
    ids=["only-attributes", "header-only", "empty-file"],
)
def test_unusable_files_fail_with_a_clear_message(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        read_csv(write(tmp_path, text))


def test_the_report_shows_what_was_detected(tmp_path):
    data = read_csv(write(tmp_path, "Age,City,Q1\n30,,a\n,,\n40,,b\n"))

    report = format_import(data, "f.csv")

    assert "read 2 respondents x 1 questions from f.csv" in report
    assert "attributes found: age" in report  # city is empty everywhere, so it is not reported
    assert "Q1. Q1" in report


def test_the_sample_file_in_the_repo_imports():
    data = read_csv(str(SAMPLE))

    assert len(data.profiles) == 4 and len(data.questions) == 2
    assert data.profiles[3]["age"] is None
    assert data.blank_answers == 1
