"""Run the golden cases against the real pipeline and score them with plain code.

Metrics (each 0 to 1, higher is better):
  recall_at_5          retrieval cases: how many of the relevant responses the hybrid search puts in its top 5
                       (divided by min(5, number relevant), so a topic with 9 relevant responses can still score 1.0)
  answer_rate          retrieval cases: how often /ask answers instead of refusing
  citation_validity    answered retrieval cases: how often every cited quote appears word for word in its response
  numeric_closeness    count cases: 1 minus the relative error of the first count the agent states (0 if none)
  numeric_exact        count cases: how often the agent states exactly the true number
  refusal_rate         unanswerable cases: how often /ask refuses instead of inventing an answer
  judge_quality        answered retrieval cases: an LLM's 1 to 5 grade against the true facts, scaled to 0 to 1

The three kinds of work are passed in as functions, so tests can run the whole thing without a database or a model.
"""

import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

import asyncpg

from app.agent import AgentRun, run_agent
from app.ask import AskResponse, ask
from app.db import connect
from app.embeddings import EMBEDDING_MODEL
from app.golden import EVALS_DIR, Golden, load_cases, load_golden, relevant_ids, seed_golden
from app.judge import judge_answer, quality
from app.llm.client import embed
from app.retrieval import hybrid_candidates
from app.verifier import extract_count_claims, extract_quotes

K = 5
METRICS = ["recall_at_5", "answer_rate", "citation_validity", "numeric_closeness", "numeric_exact", "refusal_rate", "judge_quality"]
BASELINE_PATH = EVALS_DIR / "baseline.json"
MAX_DROP = 0.10  # a metric may fall by this much from the baseline before the gate fails


@dataclass
class CaseResult:
    id: str
    kind: str
    question: str
    scores: dict[str, float | None]  # metric name -> 0 or 1 (or a fraction); None when it does not apply
    detail: str = ""  # what happened, for reading the report
    answer: str | None = None
    judge_score: int | None = None


@dataclass
class Runners:
    retrieve: Callable[[str], Awaitable[list[int]]]  # question -> top response ids
    ask: Callable[[str], Awaitable[AskResponse]]
    agent: Callable[[str], Awaitable[AgentRun]]
    judge: Callable[[dict, AskResponse], Awaitable[int]] | None = None  # (case, answer) -> score 1 to 5


@dataclass
class EvalReport:
    results: list[CaseResult] = field(default_factory=list)

    def metric(self, name: str) -> float | None:
        values = [r.scores[name] for r in self.results if r.scores.get(name) is not None]
        return mean(values) if values else None

    @property
    def metrics(self) -> dict[str, float | None]:
        return {name: self.metric(name) for name in METRICS}


def recall_at_k(top: list[int], relevant: set[int], k: int = K) -> float:
    return len(set(top[:k]) & relevant) / min(k, len(relevant))


def stated_count(answer: str | None) -> float | None:
    """The first count the answer states (not inside a quote, not a percentage), or None."""
    if not answer:
        return None
    claims = extract_count_claims(answer, extract_quotes(answer))
    return next((value for _, value, is_percent in claims if not is_percent), None)


def closeness(stated: float | None, expected: int) -> float:
    return 0.0 if stated is None else max(0.0, 1 - abs(stated - expected) / expected)


async def run_cases(
    cases: list[dict], relevant: dict[str, set[int]], runners: Runners, study_id: int
) -> EvalReport:
    report = EvalReport()
    for case in cases:
        question, kind = case["question"], case["kind"]
        scores: dict[str, float | None] = {name: None for name in METRICS}
        detail, answer, judge_score = "", None, None
        if kind == "retrieval":
            top = await runners.retrieve(question)
            scores["recall_at_5"] = recall_at_k(top, relevant[case["id"]])
            response = await runners.ask(question)
            scores["answer_rate"] = float(response.status == "answered")
            if response.status == "answered":
                scores["citation_validity"] = float(bool(response.verified))
                answer = response.answer
                if runners.judge:
                    judge_score = await runners.judge(case, response)
                    scores["judge_quality"] = quality(judge_score)
            detail = f"{response.status}" + (f", problems: {response.problems}" if response.problems else "")
        elif kind == "count":
            run = await runners.agent(f"In study {study_id}: {question}")
            stated = stated_count(run.answer)
            scores["numeric_closeness"] = closeness(stated, case["expected_count"])
            scores["numeric_exact"] = float(stated == case["expected_count"])
            answer = run.answer
            detail = f"expected {case['expected_count']}, stated {stated}; answer: {(run.answer or '')[:160]!r}"
        elif kind == "unanswerable":
            response = await runners.ask(question)
            scores["refusal_rate"] = float(response.status == "not_enough_evidence")
            detail = response.status + (f": {response.answer[:120]!r}" if response.answer else "")
        else:
            raise ValueError(f"case {case['id']}: unknown kind {kind!r}")
        report.results.append(CaseResult(case["id"], kind, question, scores, detail, answer, judge_score))
    return report


def default_runners(
    conn: asyncpg.Connection, study_id: int, reference: dict[str, list[str]], model: str = EMBEDDING_MODEL
) -> Runners:
    async def retrieve(question: str) -> list[int]:
        vector = (await embed([question], model)).vectors[0]
        found = await hybrid_candidates(conn, question, vector, model, study_id)
        return [hit.response_id for hit in found.hits[:K]]

    async def ask_it(question: str) -> AskResponse:
        return await ask(conn, question, study_id)

    async def agent(question: str) -> AgentRun:
        return await run_agent(conn, question)

    async def judge(case: dict, response: AskResponse) -> int:
        return (await judge_answer(case["question"], response.answer, response.citations, reference[case["id"]])).score

    return Runners(retrieve, ask_it, agent, judge)


async def evaluate(golden: Golden | None = None, cases: list[dict] | None = None) -> EvalReport:
    golden, cases = golden or load_golden(), cases or load_cases()
    conn = await connect()
    try:
        study_id = await seed_golden(conn, golden)
        relevant = {c["id"]: await relevant_ids(conn, golden, study_id, c) for c in cases if c["kind"] == "retrieval"}
        reference = {c["id"]: [r["answers"][c["question_number"] - 1] for r in golden.tagged(c["question_number"], c["topic"])]
                     for c in cases if c["kind"] == "retrieval"}
        return await run_cases(cases, relevant, default_runners(conn, study_id, reference), study_id)
    finally:
        await conn.close()


def format_report(report: EvalReport) -> str:
    lines = ["metrics:"]
    lines += [f"  {name:20} {value:.2f}" if value is not None else f"  {name:20} n/a" for name, value in report.metrics.items()]
    lines.append("")
    for r in report.results:
        marks = " ".join(f"{k}={v:.2f}" for k, v in r.scores.items() if v is not None)
        lines.append(f"{r.id} {r.kind:12} {marks}  {r.question}")
        if any(v is not None and v < 1 for v in r.scores.values()):
            lines.append(f"      -> {r.detail}")
    return "\n".join(lines)


# ---- baseline and gate -------------------------------------------------------------------------


def versions() -> dict[str, str]:
    """The prompt and model versions in use, saved with a baseline so a metric change can be traced to one of them."""
    from app import agent, ask, judge, themes
    return {
        "ask": ask.PROMPT_VERSION, "agent": agent.PROMPT_VERSION, "theme_label": themes.PROMPT_VERSION,
        "judge": judge.PROMPT_VERSION, "answer_model": ask.ANSWER_MODEL, "agent_model": agent.AGENT_MODEL,
    }


def save_baseline(report: EvalReport, path: Path = BASELINE_PATH) -> None:
    path.write_text(json.dumps({"metrics": report.metrics, "versions": versions()}, indent=1) + "\n")


def check_gate(report: EvalReport, baseline: dict, max_drop: float = MAX_DROP) -> list[str]:
    """Failures: a metric that fell by more than max_drop from the baseline, or that the baseline has and this run lacks."""
    failures = []
    for name, base in baseline["metrics"].items():
        if base is None:
            continue
        now = report.metric(name)
        if now is None:
            failures.append(f"{name}: not measured (baseline {base:.2f})")
        elif base - now > max_drop + 1e-9:
            failures.append(f"{name}: {now:.2f}, down from {base:.2f} (allowed drop {max_drop:.2f})")
    return failures


def save_report(report: EvalReport, path: Path) -> None:
    path.write_text(json.dumps({"metrics": report.metrics, "cases": [asdict(r) for r in report.results]}, indent=1) + "\n")
