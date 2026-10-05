import asyncio
import textwrap
from itertools import zip_longest
from statistics import mean

from app.llm.client import LLMInvalidOutput, LLMRefusal, complete_structured
from app.prompts.registry import PromptRef, build_messages
from app.schemas import PROMPT_SCHEMAS, DiscussionGuide

# A question starting with one of these can usually be answered with yes or no.
# It is a rough heuristic ("Would you describe..." is open) and a good example of
# why counting words is a weak way to judge output.
CLOSED_OPENERS = {"did", "do", "does", "is", "are", "was", "were", "have", "has", "will", "would"}


def closed_question_count(guide: DiscussionGuide) -> int:
    return sum(1 for q in guide.questions if q.text.split()[:1] and q.text.split()[0].lower() in CLOSED_OPENERS)


async def run_all(name: str, versions: list[str], cases: list[dict], runs: int, model: str) -> dict:
    """Run every version on every case `runs` times. Returns {(version, case_index): [outcome]}.

    An outcome is a guide, or the exception for a run that failed validation or was refused.
    """
    schema = PROMPT_SCHEMAS[name]
    keys = [(v, i) for v in versions for i in range(len(cases)) for _ in range(runs)]

    async def one(version: str, case: dict):
        prompt = PromptRef(name, version, case)
        return await complete_structured(build_messages(name, version, case), model, schema, prompt)

    outcomes = await asyncio.gather(*(one(v, cases[i]) for v, i in keys), return_exceptions=True)
    results: dict = {}
    for key, outcome in zip(keys, outcomes):
        if isinstance(outcome, Exception) and not isinstance(outcome, (LLMInvalidOutput, LLMRefusal)):
            raise outcome  # a bug, not a bad model answer
        results.setdefault(key, []).append(outcome)
    return results


def summarize(outcomes: list) -> dict:
    """Average the metrics over the successful runs of one (version, case) cell."""
    guides = [o for o in outcomes if not isinstance(o, Exception)]
    return {
        "closed": mean(closed_question_count(g) for g in guides) if guides else None,
        "questions": mean(len(g.questions) for g in guides) if guides else None,
        "failed": len(outcomes) - len(guides),
    }


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def format_report(name, versions, cases, runs, results, show_text=False) -> str:
    a, b = versions
    cells = {(v, i): summarize(results[(v, i)]) for v in versions for i in range(len(cases))}
    lines = [
        f"{name}: {a} vs {b} on {len(cases)} cases x {runs} runs (fewer closed questions is better)",
        "",
        f"{'#':<3}{'brief':<44}{'closed ' + a:>10}{'closed ' + b:>10}{'questions ' + a:>14}{'questions ' + b:>14}",
    ]
    for i, case in enumerate(cases):
        brief = case["brief"] if len(case["brief"]) <= 40 else case["brief"][:37] + "..."
        ca, cb = cells[(a, i)], cells[(b, i)]
        lines.append(
            f"{i + 1:<3}{brief:<44}{_num(ca['closed']):>10}{_num(cb['closed']):>10}"
            f"{_num(ca['questions']):>14}{_num(cb['questions']):>14}"
        )

    def overall(version: str, metric: str) -> float | None:
        values = [cells[(version, i)][metric] for i in range(len(cases)) if cells[(version, i)][metric] is not None]
        return mean(values) if values else None

    lines.append(
        f"{'':<3}{'average':<44}{_num(overall(a, 'closed')):>10}{_num(overall(b, 'closed')):>10}"
        f"{_num(overall(a, 'questions')):>14}{_num(overall(b, 'questions')):>14}"
    )
    failed = {v: sum(cells[(v, i)]["failed"] for i in range(len(cases))) for v in versions}
    lines += ["", f"failed runs: {a} {failed[a]}, {b} {failed[b]}"]

    closed_a, closed_b = overall(a, "closed"), overall(b, "closed")
    if closed_a is not None and closed_b is not None:
        verdict = f"{b} is better" if closed_b < closed_a else f"{a} is better" if closed_a < closed_b else "no difference"
        lines.append(f"closed questions per guide: {a} {closed_a:.1f}, {b} {closed_b:.1f} -> {verdict}")

    if show_text:
        for i, case in enumerate(cases):
            lines += ["", f"--- case {i + 1}: {case['brief']}", f"{a:<76}| {b}"]
            columns = []
            for v in versions:
                guide = next((o for o in results[(v, i)] if not isinstance(o, Exception)), None)
                questions = [f"{n}. {q.text}" for n, q in enumerate(guide.questions, 1)] if guide else ["(no valid run)"]
                columns.append([line for q in questions for line in textwrap.wrap(q, 74)])
            for left, right in zip_longest(*columns, fillvalue=""):
                lines.append(f"{left:<76}| {right}")
    return "\n".join(lines)
