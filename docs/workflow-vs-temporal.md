# Our workflow engine vs Temporal

Insightloop runs a study as five steps (guide, respondents, embed, themes, synthesize) using about 400 lines in
`services/api/app/workflow.py` on top of the M11 event queue. Temporal is a purpose-built engine for the same
problem. This page says what each does and what ours lacks.

## What we built

- **State in rows.** `workflow_runs` and `workflow_steps` hold status, attempts and each step's output. A crashed run
  resumes at the first step that is not done (a checkpoint per step).
- **Delivery by the event queue.** A `workflow.advance` event is redelivered when a worker dies (lease expiry) and
  retried with backoff, then dead-lettered.
- **One executor per run** through a Postgres advisory lock.
- **Fan-out with fan-in.** Respondents are generated in parallel, each saved under a `(study, slot)` key. A resume
  redoes only the missing ones. The step passes at 80% and records the failed slots.
- **Pause, cancel and retry-from-step** as status changes noticed at safe points. Retrying from a step undoes that
  step and the later ones first (compensation).

## What Temporal gives you that ours lacks

| Need | Temporal | Ours |
|---|---|---|
| Resume inside a step | Replays the workflow's history, so code resumes at the exact line | Resumes at the step. A half-done step redoes its unfinished work (respondents are saved one by one to soften this) |
| Determinism | The engine re-runs workflow code from history, so that code must be deterministic (no clock, no random, no direct I/O) and the engine checks for violations | We never replay code. Steps are idempotent by hand (slot keys, status guards) and nothing enforces that |
| Timers and sleeps | Durable timers that survive restarts, for waits of days | Not supported. Only polling and backoff |
| Retries | Per-activity retry policy, timeouts, heartbeats | One policy for the whole event. A deterministic error (too few answers for themes) was retried five times before we added an up-front check |
| Visibility | Web UI with every run, its history and a search | `GET /workflows/{id}` and SQL |
| Versioning | Patching and versioned workflows so running runs survive a code change | None. Changing the step list under a running run is unsafe |
| Signals and queries | Send a message to or read state from a live run | Pause and cancel are polled status flags |
| Scale and ops | Sharded service with its own persistence | One Postgres and one worker process |
| Child workflows | First class | Not supported. The step list is fixed |

## What ours does not handle

- A run that needs more than one worker per run (the advisory lock allows one executor).
- A step that outlives the event lease (120 seconds). A second worker takes the event, finds the run's lock held,
  and marks the event done. If the first worker then dies, nobody is left to advance the run, which stays
  "running". The fix is a heartbeat that renews the lease while a handler works, as Temporal's activity
  heartbeats do. We have not built it.
- Undoing external side effects. Compensation only deletes our own rows. A model call cannot be undone.
- Cancel keeps partial data. There is no option to delete it.
- Changes to the step list or step code while runs are in flight.

## When to pick which

Ours is enough for a handful of fixed steps that are each safe to repeat and run for minutes. Temporal pays off
when workflows are long-lived, have waits and branches, change often, or when many teams need to see and operate
them. We have not ported a step to Temporal.
