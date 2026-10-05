import argparse
import asyncio
import json
import logging
import time

from app.agent_compare import compare_systems, format_comparison
from app.multi_agent import research_question
from app.agent import MAX_STEPS, MAX_TOKENS, agent_answer, format_agent_run
from app.ask import answer_question, format_ask
from app.chunk_experiment import chunk_compare, format_experiment
from app.compare import format_report, run_all
from app.db import connect, fetch_call
from app.embeddings import EMBEDDING_MODEL, embed_all
from app.evals import BASELINE_PATH, check_gate, evaluate, format_report, save_baseline, save_report
from app.golden import EVALS_DIR, ensure_themes, load_golden, seed_golden
from app.judge import agreement, read_labels
from app.importer import format_import, read_csv
from app.index_bench import COPIES, bench_index, format_bench
from app.retrieval_eval import format_eval, retrieval_eval
from app.llm.client import complete, complete_structured
from app.personas import DEFAULT_VERSION, format_personas, generate_personas
from app.respondents import format_simulation, simulate
from app.search import SAMPLE_QUERIES, format_neighbors, search
from app.themes import format_run, themes_for_question
from app.store import save_import, save_simulation
from app.verifier import format_verification
from app.prompts.registry import PromptRef
from app.schemas import SCHEMAS


async def _question_numbers(study_id: int) -> list[int]:
    conn = await connect()
    try:
        rows = await conn.fetch(
            """
            SELECT DISTINCT s.question_number FROM responses s
            JOIN respondents r ON r.id = s.respondent_id WHERE r.study_id = $1 ORDER BY 1
            """,
            study_id,
        )
        return [row["question_number"] for row in rows]
    finally:
        await conn.close()


async def _themes_command(args) -> None:
    questions = [args.question] if args.question else await _question_numbers(args.study)
    for number in questions:
        run, run_id = await themes_for_question(args.study, number, args.method, args.k, save=not args.no_save)
        print()
        print(format_run(run))
        print(f"  (saved as theme run {run_id})" if run_id else "  (not saved)")


class CallNotFound(Exception):
    pass


class NotReplayable(Exception):
    pass


async def replay(call_id: int) -> tuple[dict, str]:
    """Rerun a stored call with its exact model, messages and params.

    Returns the stored row and the new output. The rerun is recorded as a new row.
    """
    row = await fetch_call(call_id)
    if row is None:
        raise CallNotFound(call_id)

    messages = json.loads(row["messages"])
    params = json.loads(row["params"])
    if "tools" in params:
        raise NotReplayable(f"call {call_id} used tools ({', '.join(params['tools'])}); replay only handles plain and structured calls")
    prompt = None
    if row["prompt_name"]:
        prompt = PromptRef(row["prompt_name"], row["prompt_version"], json.loads(row["inputs"]))

    if "response_format" in params:
        schema = SCHEMAS[params["response_format"]]
        result = await complete_structured(messages, row["model"], schema, prompt)
        return row, result.model_dump_json()
    result = await complete(messages, row["model"], prompt)
    return row, result.text


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    replay_parser = commands.add_parser("replay", help="rerun a stored LLM call exactly")
    replay_parser.add_argument("call_id", type=int)
    compare_parser = commands.add_parser("compare", help="run two prompt versions over the same cases")
    compare_parser.add_argument("prompt")
    compare_parser.add_argument("version_a")
    compare_parser.add_argument("version_b")
    compare_parser.add_argument("--cases", required=True, help="JSON file: a list of input objects")
    compare_parser.add_argument("--runs", type=int, default=3, help="runs per case per version")
    compare_parser.add_argument("--model", default="gpt-4o-mini")
    compare_parser.add_argument("--text", action="store_true", help="also print the questions side by side")
    personas_parser = commands.add_parser("personas", help="generate synthetic respondents")
    personas_parser.add_argument("--brief", required=True)
    personas_parser.add_argument("--count", type=int, default=30)
    personas_parser.add_argument("--seed", type=int, default=0)
    personas_parser.add_argument("--model", default="gpt-4o-mini")
    personas_parser.add_argument("--version", default=DEFAULT_VERSION, help="persona prompt version")
    respond_parser = commands.add_parser("respond", help="simulate respondents answering a guide")
    respond_parser.add_argument("--brief", required=True)
    respond_parser.add_argument("--count", type=int, default=30)
    respond_parser.add_argument("--seed", type=int, default=0)
    respond_parser.add_argument("--questions", type=int, default=6, help="how many guide questions to ask")
    respond_parser.add_argument("--concurrency", type=int, default=10, help="max API calls in flight")
    respond_parser.add_argument("--model", default="gpt-4o-mini")
    respond_parser.add_argument(
        "--csv", help="import real responses from this file (one row per respondent) instead of simulating"
    )
    embed_parser = commands.add_parser("embed", help="embed responses that have no vector yet")
    embed_parser.add_argument("--study", type=int, help="only this study (default: every study)")
    embed_parser.add_argument("--model", default=EMBEDDING_MODEL)
    commands.add_parser("eval-seed", help="load and embed the golden study the evals run against")
    eval_parser = commands.add_parser("eval", help="run the golden cases and score them")
    eval_parser.add_argument("--save-baseline", action="store_true", help="record these metrics as the new baseline")
    eval_parser.add_argument("--gate", action="store_true", help="exit with an error if a metric dropped below the baseline")
    eval_parser.add_argument("--max-drop", type=float, default=0.10)
    eval_parser.add_argument("--sheet", action="store_true", help="write evals/hand_labels.json for you to label")
    research_parser = commands.add_parser("research", help="answer a question with planner, analysts, synthesizer and critic")
    research_parser.add_argument("question")
    worker_parser = commands.add_parser("worker", help="run the background worker that handles queued events")
    worker_parser.add_argument("--lease", type=float, default=120, help="seconds a claimed event stays locked to a worker")
    worker_parser.add_argument("--once", action="store_true", help="handle events until none are left, then exit")
    submit_parser = commands.add_parser("submit", help="queue a research question for the worker")
    submit_parser.add_argument("question")
    request_parser = commands.add_parser("request", help="show a queued research request")
    request_parser.add_argument("id", type=int)
    flow_parser = commands.add_parser("workflow", help="start, inspect or control a durable study run")
    flow_parser.add_argument("action", choices=["start", "show", "pause", "resume", "cancel", "retry"])
    flow_parser.add_argument("target", help="the brief (for start) or the run id")
    flow_parser.add_argument("--respondents", type=int, default=10)
    flow_parser.add_argument("--from-step", default=None)
    commands.add_parser("agent-compare", help="single agent against multi-agent on the golden cases")
    commands.add_parser("eval-agreement", help="how often the LLM judge agrees with your hand labels")
    neighbors_parser = commands.add_parser("neighbors", help="show the nearest responses to some queries")
    neighbors_parser.add_argument("queries", nargs="*", help="default: a few sample queries")
    neighbors_parser.add_argument("--k", type=int, default=5)
    neighbors_parser.add_argument("--model", default=EMBEDDING_MODEL)
    bench_parser = commands.add_parser("index-bench", help="compare exact search, HNSW and IVFFlat")
    bench_parser.add_argument("--copies", type=int, default=COPIES, help="noisy copies of each real vector")
    bench_parser.add_argument("--model", default=EMBEDDING_MODEL)
    chunk_parser = commands.add_parser("chunk-compare", help="compare whole-answer and chunked embeddings")
    chunk_parser.add_argument("--model", default=EMBEDDING_MODEL)
    ask_parser = commands.add_parser("ask", help="ask a question about the stored responses")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--study", type=int, help="only this study (default: all)")
    ask_parser.add_argument("--k", type=int, default=8, help="how many responses to retrieve")
    ask_parser.add_argument("--mode", choices=["vector", "hybrid"], default="hybrid")
    ask_parser.add_argument("--rerank", action="store_true", help="re-score the candidates with the model")
    eval_parser = commands.add_parser("retrieval-eval", help="compare vector, keyword, hybrid and rerank")
    eval_parser.add_argument("--no-rerank", action="store_true", help="skip the rerank column (it makes LLM calls)")
    themes_parser = commands.add_parser("themes", help="find and name the themes in a study's answers")
    themes_parser.add_argument("--study", type=int, required=True)
    themes_parser.add_argument("--question", type=int, help="question number (default: every question)")
    themes_parser.add_argument("--method", choices=["kmeans", "hdbscan"], default="kmeans")
    themes_parser.add_argument("--k", type=int, help="number of themes for kmeans (default: chosen by silhouette)")
    themes_parser.add_argument("--no-save", action="store_true", help="show the themes without storing them")
    agent_parser = commands.add_parser("agent", help="answer a question by letting the model use tools")
    agent_parser.add_argument("question")
    agent_parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    agent_parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args()

    if args.command == "agent":
        print()
        print(format_agent_run(asyncio.run(agent_answer(args.question, args.max_steps, args.max_tokens))))

    if args.command == "themes":
        asyncio.run(_themes_command(args))  # one event loop for every question: the OpenAI client is tied to its loop

    if args.command == "retrieval-eval":
        print()
        print(format_eval(asyncio.run(retrieval_eval(use_rerank=not args.no_rerank))))

    if args.command == "ask":
        print()
        print(format_ask(asyncio.run(answer_question(args.question, args.study, args.k, args.mode, args.rerank))))

    if args.command == "chunk-compare":
        report = asyncio.run(chunk_compare(args.model))
        print()
        print(format_experiment(report))

    if args.command == "index-bench":
        report = asyncio.run(bench_index(args.model, args.copies))
        print()
        print(format_bench(report))

    if args.command == "neighbors":
        results = asyncio.run(search(args.queries or SAMPLE_QUERIES, args.model, args.k))
        print()
        print(format_neighbors(results))

    if args.command == "eval-seed":

        async def seed() -> int:
            conn = await connect()
            try:
                golden = load_golden()
                study_id = await seed_golden(conn, golden)
                return study_id, await ensure_themes(conn, study_id, len(golden.questions))
            finally:
                await conn.close()

        study_id, built = asyncio.run(seed())
        print(f"golden study ready: study {study_id} ({built} theme runs built)")

    if args.command == "eval":
        report = asyncio.run(evaluate())
        print(format_report(report))
        save_report(report, EVALS_DIR / "last_run.json")
        if args.sheet:
            rows = [
                {"case_id": r.id, "question": r.question, "answer": r.answer, "judge_score": r.judge_score, "human_score": None}
                for r in report.results if r.judge_score is not None
            ][:10]
            (EVALS_DIR / "hand_labels.json").write_text(json.dumps(rows, indent=1) + "\n")
            print(f"\nwrote {len(rows)} answers to evals/hand_labels.json: fill in human_score (1 to 5), then run make eval-agreement")
        if args.save_baseline:
            save_baseline(report)
            print(f"\nbaseline saved to {BASELINE_PATH}")
        if args.gate:
            failures = check_gate(report, json.loads(BASELINE_PATH.read_text()), args.max_drop)
            if failures:
                print("\nGATE FAILED:\n  " + "\n  ".join(failures))
                raise SystemExit(1)
            print("\ngate passed")

    if args.command == "research":
        result = asyncio.run(research_question(args.question))
        print("plan:\n" + "\n".join(f"  - {q}" for q in result.plan))
        for finding in result.findings:
            print(f"\nanalyst (run {finding.run_id}, {finding.status}): {finding.sub_question}\n  {finding.answer}")
        for verdict in result.critiques:
            print("\ncritic: " + ("no gaps" if verdict.ok else "gaps: " + "; ".join(verdict.gaps)))
        print(f"\nreport ({result.seconds:.0f}s):\n{result.answer}\n\n{format_verification(result.verification)}")

    if args.command == "worker":
        from app.worker import run_worker, work_once

        async def drain() -> int:
            conn = await connect()
            try:
                handled = 0
                while await work_once(conn, lease_seconds=args.lease):
                    handled += 1
                return handled
            finally:
                await conn.close()

        if args.once:
            print(f"handled {asyncio.run(drain())} events")
        else:
            print("worker running, Ctrl+C to stop")
            try:
                asyncio.run(run_worker(lease_seconds=args.lease))
            except KeyboardInterrupt:
                pass

    if args.command in ("submit", "request"):
        from app import research_jobs

        async def show() -> None:
            conn = await connect()
            try:
                if args.command == "submit":
                    print(f"queued research request {await research_jobs.submit(conn, args.question)}")
                else:
                    row = await research_jobs.get(conn, args.id)
                    print(dict(row) if row else f"no request {args.id}")
            finally:
                await conn.close()

        asyncio.run(show())

    if args.command == "workflow":
        from app import workflow

        async def control() -> None:
            conn = await connect()
            try:
                if args.action == "start":
                    if args.respondents < workflow.MIN_RESPONDENTS:
                        print(f"need at least {workflow.MIN_RESPONDENTS} respondents (the themes step needs 8 answers per question)")
                        return
                    print(f"started workflow run {await workflow.start(conn, args.target, args.respondents, 0, 'gpt-4o-mini')}")
                    return
                run_id = int(args.target)
                if args.action != "show":
                    action = {"pause": workflow.pause, "resume": workflow.resume, "cancel": workflow.cancel}.get(args.action)
                    await (action(conn, run_id) if action else workflow.retry(conn, run_id, args.from_step))
                info = await workflow.describe(conn, run_id)
                if info is None:
                    print(f"no run {run_id}")
                    return
                print(f"run {info['id']}: {info['status']}, study {info['study_id']}, {info['respondent_count']} respondents" + (f", error: {info['error']}" if info["error"] else ""))
                for step in info["steps"]:
                    detail = f" {step['output']}" if step["output"] else ""
                    print(f"  {step['name']:12} {step['status']:8} attempts={step['attempts']}{detail}{' ERROR ' + step['error'] if step['error'] else ''}")
                if info["report"]:
                    print("\nreport:\n" + info["report"])
            except workflow.WorkflowError as error:
                print(f"cannot do that: {error}")
            finally:
                await conn.close()

        asyncio.run(control())

    if args.command == "agent-compare":
        print(format_comparison(asyncio.run(compare_systems())))

    if args.command == "eval-agreement":
        pairs = read_labels(EVALS_DIR / "hand_labels.json")
        if not pairs:
            raise SystemExit("no human_score filled in yet in evals/hand_labels.json")
        result = agreement(pairs)
        print(f"{result['n']} labeled answers: exact {result['exact']:.0%}, within 1 point {result['within_1']:.0%}, same pass/fail (4 or more) {result['pass_fail']:.0%}")

    if args.command == "embed":
        summary = asyncio.run(embed_all(args.model, args.study))
        cost = f"${summary.cost_usd:.6f}" if summary.cost_usd is not None else "unknown cost"
        print(
            f"embedded {summary.embedded} responses with {args.model}: "
            f"{summary.cache_hits} texts from the cache, {summary.api_texts} sent to the API "
            f"in {summary.requests} requests ({summary.tokens} tokens, {cost})"
        )

    if args.command == "respond" and args.csv:
        try:
            data = read_csv(args.csv)
        except (OSError, ValueError) as error:
            raise SystemExit(f"Could not import {args.csv}: {error}")
        print(format_import(data, args.csv))
        study_id = asyncio.run(save_import(args.brief, data))
        print(f"\nsaved as study {study_id}")
    elif args.command == "respond":
        start = time.perf_counter()
        sim = asyncio.run(
            simulate(
                args.brief, args.count, args.seed, args.model, args.concurrency, args.questions
            )
        )
        print()
        print(format_simulation(sim, time.perf_counter() - start))
        study_id = asyncio.run(save_simulation(args.brief, args.seed, args.model, sim))
        print(f"\nsaved as study {study_id}")

    if args.command == "personas":
        personas = asyncio.run(
            generate_personas(args.brief, args.count, args.seed, args.model, args.version)
        )
        print()
        print(format_personas(personas))

    if args.command == "compare":
        with open(args.cases) as f:
            cases = json.load(f)
        versions = [args.version_a, args.version_b]
        results = asyncio.run(run_all(args.prompt, versions, cases, args.runs, args.model))
        print()
        print(format_report(args.prompt, versions, cases, args.runs, results, args.text))

    if args.command == "replay":
        try:
            row, new_output = asyncio.run(replay(args.call_id))
        except CallNotFound:
            raise SystemExit(f"No call with id {args.call_id} in llm_calls.")
        except NotReplayable as error:
            raise SystemExit(str(error))
        print(f"\nreplayed call {row['id']} ({row['prompt_name']} {row['prompt_version']}, {row['model']})")
        print(f"\n--- original ---\n{row['output']}")
        print(f"\n--- replay ---\n{new_output}")
        print(f"\nidentical: {'yes' if new_output == row['output'] else 'no'}")


if __name__ == "__main__":
    main()
