import csv
from dataclasses import dataclass

# Columns with these names describe the respondent. Every other column is a question.
ATTRIBUTES = ["name", "age", "gender", "city", "income", "household", "occupation"]
# Common export noise that is neither an attribute nor a question.
IGNORED = {"id", "respondent_id", "response_id", "timestamp"}


@dataclass
class ImportedData:
    questions: list[str]
    profiles: list[dict]
    answers: list[list[str | None]]  # answers[i][j] is respondent i's answer to question j, or None

    @property
    def blank_answers(self) -> int:
        return sum(answer is None for row in self.answers for answer in row)


def read_csv(path: str) -> ImportedData:
    """Read a survey export with one row per respondent and one column per question."""
    with open(path, newline="", encoding="utf-8-sig") as file:  # utf-8-sig drops Excel's BOM
        reader = csv.DictReader(file)
        header = [h for h in (reader.fieldnames or []) if h and h.strip()]
        attribute_columns = {h: h.strip().lower() for h in header if h.strip().lower() in ATTRIBUTES}
        question_columns = [
            h for h in header
            if h not in attribute_columns and h.strip().lower().replace(" ", "_") not in IGNORED
        ]
        if not question_columns:
            raise ValueError("No question columns found. Every column is an attribute or ignored.")

        profiles, answers = [], []
        for line_number, row in enumerate(reader, start=2):  # line 1 is the header
            if not any((value or "").strip() for value in row.values() if isinstance(value, str)):
                continue  # a blank line
            profile = {}
            for column, attribute in attribute_columns.items():
                value = (row.get(column) or "").strip() or None
                if attribute == "age" and value is not None:
                    try:
                        value = int(value)
                    except ValueError:
                        raise ValueError(f"Line {line_number}: age {value!r} is not a whole number.")
                profile[attribute] = value
            profiles.append(profile)
            answers.append([(row.get(q) or "").strip() or None for q in question_columns])

    if not profiles:
        raise ValueError("The file has a header but no respondent rows.")
    return ImportedData([q.strip() for q in question_columns], profiles, answers)


def format_import(data: ImportedData, path: str) -> str:
    attributes = [a for a in ATTRIBUTES if any(p.get(a) is not None for p in data.profiles)]
    lines = [
        f"read {len(data.profiles)} respondents x {len(data.questions)} questions from {path}",
        f"attributes found: {', '.join(attributes) or 'none'}",
        "questions:",
        *[f"  Q{n}. {q}" for n, q in enumerate(data.questions, 1)],
    ]
    if data.blank_answers:
        lines.append(f"{data.blank_answers} blank answers will be skipped")
    return "\n".join(lines)
