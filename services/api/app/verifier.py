"""Check an answer against the data with plain code, not with another model.

Quotes: a quote cited as [R123] must appear word for word in response 123. A quote with no citation must
appear in something the agent retrieved. Counts: every count or percentage claim must match a number in
a tool result, or a percentage worked out from two of them.

What this cannot catch: a true number or quote used for the wrong claim (a percentage for everyone
presented as a percentage for young respondents), a claim built from real quotes that they do not
support, and any claim with no number or quote in it.
"""

import json
import re
from dataclasses import asdict, dataclass, field

import asyncpg

from app.schemas import Citation

MIN_WORDS_UNCITED = 4  # a shorter quoted phrase with no citation is treated as a term, not as a claim
PERCENT_TOLERANCE = 1.0  # percentage points, compared with the exact value (30.0 for 9 of 30), to allow for rounding

_TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})
_QUOTE = re.compile(r'["“]([^"“”]{3,}?)["”]')
_CITATION = re.compile(r"\s*\[R(\d+(?:\s*,\s*R?\d+)*)\]")
_ALL_CITATIONS = re.compile(r"\[R[\d,\sR]+\]")

NUMBER_WORDS = {
    word: number for number, word in enumerate(
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen twenty".split(), start=1
    )
}
_COUNT_NOUN = r"(?:respondents?|responses?|answers?|people|participants?|customers?|runners?|users?|themes?|women|men)"
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
_COUNT = re.compile(
    rf"\b(\d+|{'|'.join(NUMBER_WORDS)})\s+(?:(?:more|other|younger|older|different|total|distinct|separate)\s+)?{_COUNT_NOUN}\b",
    re.IGNORECASE,
)
_OF = re.compile(r"\b(\d+)\s+(?:out\s+of|of)\s+(?:the\s+|all\s+)?(\d+)\b", re.IGNORECASE)
# A number right after one of these names a study, a question or an age, not a count ("question 3 respondents").
_IDENTIFIER_BEFORE = re.compile(r"(?:study|question|q|run|theme|ages?|aged|under|over|than|between)\s*$", re.IGNORECASE)
_IGNORED_KEYS = {"response_id", "study_id", "question_number", "age", "theme_id", "run_id", "id"}


@dataclass
class Problem:
    kind: str  # "quote" or "number"
    claim: str
    detail: str


@dataclass
class Verification:
    verified: bool
    problems: list[Problem] = field(default_factory=list)
    quotes_checked: int = 0
    numbers_checked: int = 0

    def as_json(self) -> list[dict]:
        return [asdict(problem) for problem in self.problems]


@dataclass
class Quote:
    text: str
    response_ids: list[int]  # empty when the quote has no citation
    start: int
    end: int


# ---- quotes ------------------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Collapse whitespace and make curly quotes and dashes plain, so only the words are compared."""
    return " ".join(text.translate(_TYPOGRAPHY).split())


def _find(piece: str, haystack: str, start: int) -> re.Match | None:
    """Find the piece, letting the case of its first letter differ ("It just..." quoted as "it just...")."""
    first = piece[0]
    return re.search(f"[{re.escape(first.lower())}{re.escape(first.upper())}]{re.escape(piece[1:])}", haystack[start:])


def quote_in_text(quote: str, text: str) -> bool:
    """Whether the quote appears word for word. "..." inside a quote means skipped text: each piece must
    appear in order. Punctuation at the edges of a piece is ignored, because it often sits outside the quote,
    and the first letter of a piece may differ in case, because a quote is often lowercased to fit a sentence.
    Everything else must match exactly, including case."""
    pieces = [piece.strip(" .,;:!?-'\"") for piece in re.split(r"\.\.\.|…", normalize(quote))]
    pieces = [piece for piece in pieces if piece]
    haystack, position = normalize(text), 0
    for piece in pieces:
        found = _find(piece, haystack, position)
        if found is None:
            return False
        position += found.end()
    return bool(pieces)


def extract_quotes(answer: str) -> list[Quote]:
    quotes = []
    for match in _QUOTE.finditer(answer):
        cited = _CITATION.match(answer, match.end())
        ids = [int(i) for i in re.findall(r"\d+", cited.group(1))] if cited else []
        quotes.append(Quote(match.group(1), ids, match.start(), match.end()))
    return quotes


# ---- numbers -----------------------------------------------------------------------------------


@dataclass
class NumberPool:
    counts: set[float] = field(default_factory=set)  # numbers that appear in tool results
    percents: set[float] = field(default_factory=set)  # percentages worked out from them, not rounded


def _collect(node, pool: NumberPool, total=None, key=None) -> None:
    if isinstance(node, dict):
        if isinstance(node.get("responses"), int):
            total = node["responses"]
        if isinstance(node.get("share"), (int, float)):
            pool.percents.add(100 * node["share"])
        size = node.get("size")
        if isinstance(size, int) and total:
            pool.percents.add(100 * size / total)
        if isinstance(node.get("counts"), dict) and isinstance(size, int) and size:
            pool.percents.update(100 * n / size for n in node["counts"].values() if isinstance(n, int))
        totals = node.get("totals")
        if isinstance(totals, dict):
            if total:
                pool.percents.update(100 * n / total for n in totals.values() if isinstance(n, int))
            for theme in node.get("themes", []):
                for group, n in (theme.get("counts") or {}).items():
                    if totals.get(group):
                        pool.percents.add(100 * n / totals[group])
        for child_key, child in node.items():
            _collect(child, pool, total, child_key)
    elif isinstance(node, list):
        for item in node:
            _collect(item, pool, total, key)
    elif isinstance(node, (int, float)) and not isinstance(node, bool) and key not in _IGNORED_KEYS:
        pool.counts.add(float(node))


def build_pool(tool_results: list[dict]) -> NumberPool:
    pool = NumberPool()
    for result in tool_results:
        _collect(result, pool)
    return pool


def extract_count_claims(answer: str, quotes: list[Quote]) -> list[tuple[str, float, bool]]:
    """(text, value, is_percent) for each count or percentage the answer states.

    Numbers inside quotes and in [R123] citations are not claims. Identifiers like "study 45" or "ages
    20 to 29" are not counts, because only a number before a counting word, or a percentage, counts.
    """
    masked = list(answer)
    for quote in quotes:
        masked[quote.start : quote.end] = " " * (quote.end - quote.start)
    text = _ALL_CITATIONS.sub(lambda m: " " * len(m.group(0)), "".join(masked))

    claims: dict[tuple[float, bool], str] = {}
    for match in _PERCENT.finditer(text):
        claims.setdefault((float(match.group(1)), True), match.group(0))
    for match in _COUNT.finditer(text):
        if _IDENTIFIER_BEFORE.search(text[: match.start()]):
            continue
        raw = match.group(1).lower()
        value = float(NUMBER_WORDS[raw]) if raw in NUMBER_WORDS else float(raw)
        if value != 1:  # "one respondent said" is phrasing, not a statistic
            claims.setdefault((value, False), match.group(0))
    for match in _OF.finditer(text):
        for group in (1, 2):
            claims.setdefault((float(match.group(group)), False), match.group(0))
    return [(claim, value, is_percent) for (value, is_percent), claim in claims.items()]


# ---- verifying ---------------------------------------------------------------------------------


def _strings(node) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [s for value in node.values() for s in _strings(value)]
    if isinstance(node, list):
        return [s for item in node for s in _strings(item)]
    return []


async def _also_found_in(conn: asyncpg.Connection, quote: str) -> list[int]:
    """Ids of responses that do contain the quote's longest piece, to spot a misattributed quote."""
    pieces = [p.strip(" .,;:!?-'\"") for p in re.split(r"\.\.\.|…", normalize(quote))]
    longest = max((p for p in pieces if p), key=len, default="")
    if not longest:
        return []
    rows = await conn.fetch(
        "SELECT id FROM responses WHERE position($1 in translate(answer, '’‘“”', '''''\"\"')) > 0 ORDER BY id LIMIT 3",
        longest,
    )
    return [row["id"] for row in rows]


async def verify_quotes(conn: asyncpg.Connection, quotes: list[Quote], retrieved_text: str) -> tuple[list[Problem], int]:
    problems, checked = [], 0
    for quote in quotes:
        claim = f'"{quote.text}"' + (f" [R{', R'.join(str(i) for i in quote.response_ids)}]" if quote.response_ids else "")
        if quote.response_ids:
            checked += 1
            rows = await conn.fetch("SELECT id, answer FROM responses WHERE id = ANY($1)", quote.response_ids)
            by_id = {row["id"]: row["answer"] for row in rows}
            gone = [i for i in quote.response_ids if i not in by_id]
            if gone:
                problems.append(Problem("quote", claim, f"response {', '.join(f'R{i}' for i in gone)} does not exist"))
            elif not any(quote_in_text(quote.text, text) for text in by_id.values()):
                elsewhere = await _also_found_in(conn, quote.text)
                where = f"these words appear in {', '.join(f'R{i}' for i in elsewhere)} instead" if elsewhere else "these words appear in no response"
                problems.append(Problem("quote", claim, f"not found in {', '.join(f'R{i}' for i in quote.response_ids)}; {where}"))
        elif len(quote.text.split()) >= MIN_WORDS_UNCITED:
            checked += 1
            if not quote_in_text(quote.text, retrieved_text):
                problems.append(Problem("quote", claim, "quoted without a citation, and not found in anything the agent retrieved"))
    return problems, checked


async def verify_answer(conn: asyncpg.Connection, answer: str, tool_results: list[str]) -> Verification:
    """Check an agent's final answer against the tool results it was given and the stored responses."""
    parsed = []
    for text in tool_results:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not (isinstance(value, dict) and "error" in value):
            parsed.append(value)

    quotes = extract_quotes(answer)
    problems, quotes_checked = await verify_quotes(conn, quotes, " ".join(s for p in parsed for s in _strings(p)))

    pool, numbers_checked = build_pool(parsed), 0
    for claim, value, is_percent in extract_count_claims(answer, quotes):
        numbers_checked += 1
        if is_percent:
            if not any(abs(value - p) <= PERCENT_TOLERANCE for p in pool.percents):
                problems.append(Problem("number", claim, "no share or count in the tool results gives this percentage"))
        elif value not in pool.counts:
            problems.append(Problem("number", claim, f"no tool result contains the number {value:g}"))
    return Verification(not problems, problems, quotes_checked, numbers_checked)


async def verify_citations(conn: asyncpg.Connection, citations: list[Citation]) -> Verification:
    """For /ask: each citation's quote must appear in the response it names."""
    quotes = [Quote(c.quote, [c.response_id], 0, 0) for c in citations]
    problems, checked = await verify_quotes(conn, quotes, "")
    return Verification(not problems, problems, checked, 0)


def format_verification(verification: Verification) -> str:
    checked = f"{verification.quotes_checked} quotes and {verification.numbers_checked} numbers checked"
    if verification.verified:
        return f"verified ({checked})"
    lines = [f"UNVERIFIED ({checked}):"]
    lines += [f"  - {p.kind}: {p.claim}: {p.detail}" for p in verification.problems]
    return "\n".join(lines)
