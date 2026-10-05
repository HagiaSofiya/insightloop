# Insightloop

Insightloop turns a research brief into themes and a cited report. It writes a discussion guide, interviews synthetic respondents, clusters their answers into themes and writes a report in which every quote and number is checked against the data.

## What it does

1. **Guide.** A brief goes in, a discussion guide comes out. You can edit the questions before anything runs.
2. **Respondents.** Synthetic personas are sampled by code (so age, income and city are spread evenly) and interviewed by a model.
3. **Embed and cluster.** Answers become vectors. Each question's answers are clustered into themes, and each theme is named by the model.
4. **Report.** A planner splits the question, analysts research the parts with tools in parallel, a synthesizer writes the report and a critic looks for gaps.
5. **Verify.** Plain code checks that every quote appears word for word in the response it cites and that every count matches a tool result.

You can also import real survey responses from a CSV and run the same analysis on them.

## Features

- **Hybrid retrieval.** pgvector (HNSW, cosine) plus keyword search, fused with reciprocal rank fusion. Questions the data cannot answer are refused instead of invented.
- **Tool-using agent.** Four tools (search, themes, counts by group, quotes) with step and token budgets. Tool errors are fed back to the model.
- **Multi-agent pipeline.** Planner, analysts, synthesizer and critic pass typed objects, not chat history.
- **Durable workflow.** A study run is a state machine saved in Postgres. It resumes after a crash and supports pause, cancel and retry from a step.
- **Event queue.** An outbox table and a worker (`SELECT ... FOR UPDATE SKIP LOCKED`) with retries, backoff and a dead-letter state.
- **Live updates.** Server-Sent Events with `Last-Event-ID` replay. The report streams token by token.
- **Web UI.** Next.js and shadcn/ui: start a study, review the guide, watch progress and cost, read insights, click a citation to see the original answer and open a "show reasoning" panel.
- **Evals.** A fixed fake dataset with known answers, five metrics, an LLM judge, a saved baseline and a regression gate.
- **Cost tracking.** Every model call is logged with model, tokens, latency and cost, and tied to the run that caused it.
- **Versioned prompts.** Prompts are immutable files under `services/api/app/prompts/`, locked by a hash manifest.

## Architecture

```
Browser (Next.js)
   |  REST + SSE
FastAPI  ----writes request + event, one transaction---->  Postgres 16 + pgvector
                                                              ^        |
Worker  --claims events, runs workflow steps, saves state-----+        |
   |                                                                   |
   +--> OpenAI API (chat, embeddings)        events, runs, steps, responses,
                                             vectors, themes, llm_calls ledger
```

## Requirements

- Docker (for Postgres)
- Python 3.11 or newer
- Node.js 20 or newer
- An OpenAI API key

## Setup

```bash
# 1. Secrets. The key is read from .env, which is git-ignored.
echo "OPENAI_API_KEY=your-key-here" > .env

# 2. Backend
python3 -m venv services/api/.venv
services/api/.venv/bin/pip install -r services/api/requirements-dev.txt

# 3. Database (Postgres on host port 5433)
make up
make db

# 4. Frontend
cd services/web && npm install
```

## Run it

Start each in its own terminal, from the repo root:

```bash
make api      # API on http://localhost:8000 (interactive docs at /docs)
make worker   # background worker: without it, runs never advance
make web      # UI on http://localhost:3100
```

Open http://localhost:3100, describe what you want to learn and start a study. A 10-respondent study takes one to two minutes and costs about two cents with `gpt-4o-mini`.

Stop everything with `Ctrl+C` in each terminal and `make down` for the database.

## Common commands

| Command | What it does |
|---|---|
| `make test` | Run the test suite. No test calls the real model. |
| `make eval-seed` | Load the fixed evaluation dataset and build its themes |
| `make eval` | Score the pipeline on the golden cases |
| `make eval ARGS="--gate"` | Same, and fail if a metric dropped below the baseline |
| `make ask ARGS="your question"` | Ask a question of the stored responses |
| `make agent ARGS="your question"` | Run the tool-using agent on a question |
| `make research Q="..."` | Run the planner, analysts, synthesizer and critic |
| `make submit Q="..."` | Queue a research question for the worker |
| `make flow ACTION=start TARGET="a brief"` | Start a durable study run from the CLI |
| `make lock-prompts` | Record the hash of a new prompt version |

The full list of commands is in the `Makefile`.

## API

| Endpoint | Purpose |
|---|---|
| `POST /workflows` | Start a study run |
| `GET /workflows`, `GET /workflows/{id}` | List runs, or show one with its steps and cost |
| `POST /workflows/{id}/pause`, `resume`, `cancel`, `retry` | Control a run |
| `POST /workflows/{id}/guide` | Save an edited guide and continue |
| `GET /workflows/{id}/events` | Live events (SSE, resumes from `Last-Event-ID`) |
| `GET /studies/{id}/insights` | Themes with counts and example answers |
| `GET /responses/{id}` | The source behind a citation |
| `GET /agent-runs/{id}` | An analyst's tool calls and verifier result |
| `POST /ask` | Answer one question from retrieved evidence, with citations |
| `POST /research`, `GET /research/{id}` | Queue a research question and read the result |

## Evaluation

`evals/` holds a fixed fake study (30 respondents, 2 questions) whose true answers are known, 19 test cases and a saved baseline. The metrics are retrieval recall at 5, answer rate, citation validity, numeric closeness, refusal on unanswerable questions and an LLM-judged quality score. `.github/workflows/evals.yml` runs the gate on pull requests that touch prompts or agent code. It needs an `OPENAI_API_KEY` repository secret.

## Project layout

```
services/api/       FastAPI backend, worker, tests, schema.sql
  app/prompts/      versioned prompt files and their hash manifest
services/web/       Next.js + shadcn/ui frontend
evals/              golden dataset, cases, baseline
docs/               design notes
```

## Known limits

- There is no authentication. Anyone who can reach the API can see every run.
- Synthetic respondents are more alike than real people: similar opinions, repeated surnames and jobs.
- Theme clusters overlap (low silhouette scores), so counts per theme are approximate. In the evals, the agent's counts differ from the true counts.
- The verifier proves that quotes and numbers exist in the data. It cannot tell whether they support the claim they are attached to.
- A workflow step that runs longer than the event lease (120 seconds) can be taken over by a second worker. A heartbeat would fix this and is not built.
- Event streams poll the database every 0.25 seconds. `LISTEN/NOTIFY` would scale better.

See `docs/workflow-vs-temporal.md` for how the workflow engine compares with Temporal.
