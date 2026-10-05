"""Single agent against the multi-agent pipeline on the same golden cases: is it better, and what does it cost?

Counting questions are scored by plain code (closeness to the true number). Open questions are graded by the
LLM judge against the true facts. Cost and tokens come from the call ledger: everything logged between the
start and the end of a case.
"""

import time
from dataclasses import dataclass, field
from statistics import mean

import asyncpg

from app.agent import run_agent
from app.db import connect
from app.evals import closeness, stated_count
from app.golden import Golden, load_cases, load_golden, seed_golden
from app.judge import JUDGE_MODEL, judge_answer, quality
from app.multi_agent import research
from app.schemas import Citation
from app.verifier import extract_quotes

SYSTEMS = ["single", "multi"]


@dataclass
class Row:
    case_id: str
    system: str
    score: float  # 0 to 1: closeness for counts, judge quality for open questions
    verified: bool | None
    cost_usd: float
    tokens: int
    llm_calls: int
    seconds: float
    note: str = ""


@dataclass
class Comparison:
    rows: list[Row] = field(default_factory=list)

    def summary(self, system: str) -> dict[str, float]:
        mine = [r for r in self.rows if r.system == system]
        checked = [r for r in mine if r.verified is not None]
        return {
            "score": mean(r.score for r in mine),
            "verified": sum(bool(r.verified) for r in checked) / len(checked) if checked else float("nan"),
            "cost_usd": sum(r.cost_usd for r in mine),
            "tokens": sum(r.tokens for r in mine),
            "calls": sum(r.llm_calls for r in mine),
            "seconds": sum(r.seconds for r in mine),
        }


async def _ledger_since(conn: asyncpg.Connection, start) -> tuple[float, int, int]:
    row = await conn.fetchrow(
        "SELECT coalesce(sum(cost_usd), 0) AS cost, coalesce(sum(input_tokens + output_tokens), 0) AS tokens, count(*) AS n "
        "FROM llm_calls WHERE created_at >= $1",
        start,
    )
    return float(row["cost"]), int(row["tokens"]), int(row["n"])


async def _score(case: dict, answer: str | None, golden: Golden) -> tuple[float, str]:
    if case["kind"] == "count":
        stated = stated_count(answer)
        return closeness(stated, case["expected_count"]), f"expected {case['expected_count']}, stated {stated}"
    reference = [r["answers"][case["question_number"] - 1] for r in golden.tagged(case["question_number"], case["topic"])]
    reference += [
        r["answers"][w["question_number"] - 1] for w in case.get("also_relevant", []) for r in golden.tagged(w["question_number"], w["topic"])
    ]
    citations = [Citation(response_id=q.response_ids[0] if q.response_ids else 0, quote=q.text) for q in extract_quotes(answer or "")]
    judgement = await judge_answer(case["question"], answer or "(no answer)", citations, reference, JUDGE_MODEL)
    return quality(judgement.score), f"judge {judgement.score}/5"


async def compare_systems(counts: int = 5, open_questions: int = 5) -> Comparison:
    golden, all_cases = load_golden(), load_cases()
    cases = [c for c in all_cases if c["kind"] == "count"][:counts] + [c for c in all_cases if c["kind"] == "retrieval"][:open_questions]
    conn = await connect()
    comparison = Comparison()
    try:
        study_id = await seed_golden(conn, golden)
        for case in cases:
            question = f"In study {study_id}: {case['question']}"
            for system in SYSTEMS:
                start = await conn.fetchval("SELECT clock_timestamp()")
                clock = time.perf_counter()
                if system == "single":
                    run = await run_agent(conn, question)
                    answer, verified = run.answer, run.verified
                else:
                    result = await research(conn, question)
                    answer, verified = result.answer, result.verification.verified
                seconds = time.perf_counter() - clock
                cost, tokens, calls = await _ledger_since(conn, start)
                score, note = await _score(case, answer, golden)
                comparison.rows.append(Row(case["id"], system, score, verified, cost, tokens, calls, seconds, note))
    finally:
        await conn.close()
    return comparison


def format_comparison(comparison: Comparison) -> str:
    lines = [f"{'case':5} {'system':7} {'score':>5} {'verified':>8} {'cost':>9} {'tokens':>7} {'calls':>5} {'secs':>5}  note"]
    for r in comparison.rows:
        verified = "-" if r.verified is None else str(r.verified)
        lines.append(f"{r.case_id:5} {r.system:7} {r.score:5.2f} {verified:>8} ${r.cost_usd:8.4f} {r.tokens:7d} {r.llm_calls:5d} {r.seconds:5.1f}  {r.note}")
    lines.append("")
    for system in SYSTEMS:
        s = comparison.summary(system)
        lines.append(
            f"{system:7} mean score {s['score']:.2f}, verified {s['verified']:.0%}, total ${s['cost_usd']:.4f}, "
            f"{s['tokens']} tokens, {s['calls']:.0f} calls, {s['seconds']:.0f}s"
        )
    return "\n".join(lines)
