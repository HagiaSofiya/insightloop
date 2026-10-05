"""Which retriever puts relevant responses in the top 5: vector, keyword, hybrid or hybrid with rerank?

A response counts as relevant if its text matches the question's pattern. The patterns are broader than
the question's own words (synonyms, word endings), but they are still keywords, so keyword search has a
built-in advantage here. Treat the result as a rough guide. M9 builds a proper golden set.
"""

import re
from dataclasses import dataclass, field
from statistics import mean

import asyncpg

from app.db import connect
from app.embeddings import EMBEDDING_MODEL
from app.llm.client import embed
from app.rerank import rerank
from app.retrieval import CANDIDATES, hybrid_candidates, hydrate, keyword_search
from app.search import Neighbor, nearest

K = 5
RETRIEVERS = ["vector", "keyword", "hybrid", "hybrid+rerank"]

# (question, a pattern that shows a response is relevant to it)
QUESTIONS = [
    ("Do respondents mention food waste?", r"\bwast(e|ed|ing)\b|thrown away|throw(ing)? .{0,15}away|\bspoil"),
    ("Is the price a reason people cancel the meal kit?", r"\bpric(e|es|ey|ing)\b|expensive|\bcost|afford"),
    ("Do people complain about the delivery?", r"\bdeliver(y|ies|ed)\b"),
    ("Do respondents worry about security or trust in banking apps?", r"\b(secur(e|ity)|trust\w*|privacy|hack\w*|fraud\w*)\b"),
    ("Do people use budgeting tools?", r"\bbudget(s|ing|ed)?\b"),
    ("Do any respondents say the app crashes or is glitchy?", r"\b(crash\w*|glitch\w*|buggy|bugs?|freez\w*)\b"),
    ("Do respondents mention knee pain?", r"\bknees?\b|\bjoints?\b"),
    ("Do runners care about arch support?", r"\barch(es)?\b"),
    ("Do runners worry about getting injured?", r"\binjur(y|ies|ed|e)\b"),
    ("Do people say running shoes are too expensive?", r"\bexpensive\b|\bpric(e|es|ey)\b|\bcost"),
]


@dataclass
class QuestionResult:
    question: str
    relevant_in_data: int
    precision: dict[str, float]  # retriever -> share of its top 5 that is relevant
    returned: dict[str, int]  # retriever -> how many it returned (rerank can return fewer than 5)


@dataclass
class EvalReport:
    results: list[QuestionResult] = field(default_factory=list)

    def mean_precision(self, retriever: str) -> float:
        return mean(r.precision[retriever] for r in self.results)

    def found_something(self, retriever: str) -> int:
        """Questions where at least one relevant response was in the top 5."""
        return sum(r.precision[retriever] > 0 for r in self.results)


def precision_at_k(hits: list[Neighbor], pattern: str, k: int = K) -> float:
    """Share of the top k that is relevant. A retriever that returns fewer than k is not rewarded for it."""
    return sum(bool(re.search(pattern, hit.answer, re.IGNORECASE)) for hit in hits[:k]) / k


async def run_eval(conn: asyncpg.Connection, model: str = EMBEDDING_MODEL, use_rerank: bool = True) -> EvalReport:
    vectors = (await embed([q for q, _ in QUESTIONS], model)).vectors
    answers = [row["answer"] for row in await conn.fetch("SELECT answer FROM responses")]  # Python's re, not Postgres's
    report = EvalReport()
    for (question, pattern), vector in zip(QUESTIONS, vectors, strict=True):
        lists: dict[str, list[Neighbor]] = {}
        lists["vector"] = await nearest(conn, vector, model, K)
        lists["keyword"] = await hydrate(conn, [i for i, _ in await keyword_search(conn, question, K)], vector, model)
        candidates = await hybrid_candidates(conn, question, vector, model)
        lists["hybrid"] = candidates.hits[:K]
        if use_rerank:
            lists["hybrid+rerank"] = await rerank(question, candidates.hits[:CANDIDATES], K)
        relevant = sum(bool(re.search(pattern, answer, re.IGNORECASE)) for answer in answers)
        report.results.append(
            QuestionResult(
                question, relevant,
                {name: precision_at_k(hits, pattern) for name, hits in lists.items()},
                {name: len(hits) for name, hits in lists.items()},
            )
        )
    return report


async def retrieval_eval(model: str = EMBEDDING_MODEL, use_rerank: bool = True) -> EvalReport:
    conn = await connect()
    try:
        return await run_eval(conn, model, use_rerank)
    finally:
        await conn.close()


def format_eval(report: EvalReport) -> str:
    names = [n for n in RETRIEVERS if n in report.results[0].precision]
    lines = [
        f"precision@{K}: the share of each retriever's top {K} that is relevant (higher is better)",
        f"{'question':<62}{'relevant':>9}" + "".join(f"{n:>15}" for n in names),
    ]
    for r in report.results:
        question = r.question if len(r.question) <= 60 else r.question[:57] + "..."
        lines.append(f"{question:<62}{r.relevant_in_data:>9}" + "".join(f"{r.precision[n]:>15.2f}" for n in names))
    lines.append(f"{'mean':<71}" + "".join(f"{report.mean_precision(n):>15.2f}" for n in names))
    lines.append(
        f"{'questions with at least one relevant result':<71}"
        + "".join(f"{report.found_something(n):>13d}/{len(report.results)}" for n in names)
    )
    return "\n".join(lines)
