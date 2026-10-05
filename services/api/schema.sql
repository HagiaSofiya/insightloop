CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS llm_calls (
    id             bigserial PRIMARY KEY,
    created_at     timestamptz NOT NULL DEFAULT now(),
    prompt_name    text,
    prompt_version text,
    inputs         jsonb,
    inputs_hash    text,
    model          text NOT NULL,
    messages       jsonb NOT NULL,
    params         jsonb NOT NULL DEFAULT '{}',
    output         text NOT NULL,
    input_tokens   integer NOT NULL,
    output_tokens  integer NOT NULL,
    latency_ms     double precision NOT NULL,
    cost_usd       numeric(12, 6)
);

CREATE TABLE IF NOT EXISTS studies (
    id         bigserial PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    brief      text NOT NULL,
    seed       integer,
    model      text
);

CREATE TABLE IF NOT EXISTS respondents (
    id             bigserial PRIMARY KEY,
    study_id       bigint NOT NULL REFERENCES studies (id) ON DELETE CASCADE,
    source         text NOT NULL CHECK (source IN ('synthetic', 'imported')),
    name           text,
    age            integer CHECK (age BETWEEN 0 AND 120),
    gender         text,
    city           text,
    income         text,
    household      text,
    occupation     text,
    backstory      text,
    speaking_style text
);
CREATE INDEX IF NOT EXISTS respondents_study_idx ON respondents (study_id);

CREATE TABLE IF NOT EXISTS responses (
    id              bigserial PRIMARY KEY,
    respondent_id   bigint NOT NULL REFERENCES respondents (id) ON DELETE CASCADE,
    question_number integer NOT NULL,
    question        text NOT NULL,
    answer          text NOT NULL,
    UNIQUE (respondent_id, question_number)
);

-- Added after the first version of the table, so they are ALTERs. The model is stored with each
-- vector because vectors from different models cannot be compared.
ALTER TABLE responses ADD COLUMN IF NOT EXISTS embedding vector(1536);
ALTER TABLE responses ADD COLUMN IF NOT EXISTS embedding_model text;

-- One vector per distinct text and model, so the same text is never embedded (and paid for) twice.
CREATE TABLE IF NOT EXISTS embedding_cache (
    content_hash text NOT NULL,
    model        text NOT NULL,
    embedding    vector(1536) NOT NULL,
    PRIMARY KEY (content_hash, model)
);

-- Pieces of a response, for the chunking experiment. Each strategy has its own set of chunks.
CREATE TABLE IF NOT EXISTS response_chunks (
    id              bigserial PRIMARY KEY,
    response_id     bigint NOT NULL REFERENCES responses (id) ON DELETE CASCADE,
    strategy        text NOT NULL,
    chunk_index     integer NOT NULL,
    text            text NOT NULL,
    embedding       vector(1536),
    embedding_model text,
    UNIQUE (response_id, strategy, chunk_index)
);

-- Approximate nearest-neighbor index for cosine distance. Postgres still chooses a plain scan
-- on tiny tables, because that is faster than walking a graph.
CREATE INDEX IF NOT EXISTS responses_embedding_hnsw ON responses USING hnsw (embedding vector_cosine_ops);

-- Vectors made before the cache existed count as cached (a no-op once they are in).
INSERT INTO embedding_cache (content_hash, model, embedding)
SELECT DISTINCT ON (encode(sha256(convert_to(answer, 'UTF8')), 'hex'), embedding_model)
       encode(sha256(convert_to(answer, 'UTF8')), 'hex'), embedding_model, embedding
FROM responses
WHERE embedding IS NOT NULL AND embedding_model IS NOT NULL
ON CONFLICT DO NOTHING;

-- Full-text search over the answers. Postgres keeps this column up to date and the GIN index makes
-- matching fast.
ALTER TABLE responses ADD COLUMN IF NOT EXISTS answer_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', answer)) STORED;
CREATE INDEX IF NOT EXISTS responses_answer_tsv_idx ON responses USING gin (answer_tsv);

-- Themes found by clustering a study's responses. Each run keeps its own themes, so runs can be compared.
CREATE TABLE IF NOT EXISTS theme_runs (
    id              bigserial PRIMARY KEY,
    study_id        bigint NOT NULL REFERENCES studies (id) ON DELETE CASCADE,
    created_at      timestamptz NOT NULL DEFAULT now(),
    method          text NOT NULL CHECK (method IN ('kmeans', 'hdbscan')),
    params          jsonb NOT NULL DEFAULT '{}',
    embedding_model text NOT NULL,
    n_responses     integer NOT NULL,
    silhouette      double precision
);

CREATE TABLE IF NOT EXISTS themes (
    id          bigserial PRIMARY KEY,
    run_id      bigint NOT NULL REFERENCES theme_runs (id) ON DELETE CASCADE,
    kind        text NOT NULL CHECK (kind IN ('theme', 'outlier')),
    label       text NOT NULL,
    description text NOT NULL,
    size        integer NOT NULL
);
CREATE INDEX IF NOT EXISTS themes_run_idx ON themes (run_id);

-- distance is how far the response is from its theme's center, so the smallest are the best examples
CREATE TABLE IF NOT EXISTS theme_members (
    theme_id    bigint NOT NULL REFERENCES themes (id) ON DELETE CASCADE,
    response_id bigint NOT NULL REFERENCES responses (id) ON DELETE CASCADE,
    distance    double precision NOT NULL,
    PRIMARY KEY (theme_id, response_id)
);

-- Which question the run clustered. Pooling questions only rediscovers the questions, so runs are per question.
ALTER TABLE theme_runs ADD COLUMN IF NOT EXISTS question_number integer;

-- One row per agent run, and one per tool call (plus one for the final answer). Steps are written as
-- they happen, so a run that crashes still leaves its trace.
CREATE TABLE IF NOT EXISTS agent_runs (
    id            bigserial PRIMARY KEY,
    created_at    timestamptz NOT NULL DEFAULT now(),
    question      text NOT NULL,
    model         text NOT NULL,
    status        text NOT NULL,
    answer        text,
    max_steps     integer NOT NULL,
    max_tokens    integer NOT NULL,
    input_tokens  integer NOT NULL DEFAULT 0,
    output_tokens integer NOT NULL DEFAULT 0,
    cost_usd      numeric(12, 6)
);

CREATE TABLE IF NOT EXISTS agent_steps (
    id            bigserial PRIMARY KEY,
    run_id        bigint NOT NULL REFERENCES agent_runs (id) ON DELETE CASCADE,
    step_number   integer NOT NULL,
    kind          text NOT NULL CHECK (kind IN ('tool', 'final')),
    tool_name     text,
    arguments     jsonb,
    result        text,
    is_error      boolean NOT NULL DEFAULT false,
    input_tokens  integer NOT NULL DEFAULT 0,
    output_tokens integer NOT NULL DEFAULT 0,
    latency_ms    double precision NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS agent_steps_run_idx ON agent_steps (run_id, step_number);

-- Whether the final answer passed the verifier, what it found wrong if not, and whether the agent got a retry.
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS verified boolean;
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS problems jsonb NOT NULL DEFAULT '[]';
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS retried boolean NOT NULL DEFAULT false;

-- M11: the event log and a first thing to put in it. An event is a row the API writes in the same
-- transaction as the state change it announces (the outbox pattern); a worker claims it later.
CREATE TABLE IF NOT EXISTS events (
    id           bigserial PRIMARY KEY,
    type         text NOT NULL,
    payload      jsonb NOT NULL DEFAULT '{}',
    created_at   timestamptz NOT NULL DEFAULT now(),
    available_at timestamptz NOT NULL DEFAULT now(),  -- not before this time (retries are pushed into the future)
    status       text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'done', 'dead')),
    attempts     integer NOT NULL DEFAULT 0,
    locked_until timestamptz,                         -- a worker's lease while status is 'processing'
    processed_at timestamptz,
    last_error   text
);
CREATE INDEX IF NOT EXISTS events_claim_idx ON events (id) WHERE status IN ('pending', 'processing');

CREATE TABLE IF NOT EXISTS research_requests (
    id         bigserial PRIMARY KEY,
    question   text NOT NULL,
    status     text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
    answer     text,
    verified   boolean,
    error      text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

-- M12: a study run as a durable state machine. Each step's state is a row, so a crashed run resumes where it stopped.
CREATE TABLE IF NOT EXISTS workflow_runs (
    id            bigserial PRIMARY KEY,
    brief         text NOT NULL,
    respondent_count integer NOT NULL CHECK (respondent_count > 0),
    seed          integer NOT NULL,
    model         text NOT NULL,
    status        text NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'running', 'paused', 'cancelled', 'failed', 'done')),
    study_id      bigint REFERENCES studies (id) ON DELETE SET NULL,
    report        text,
    error         text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS workflow_steps (
    run_id      bigint NOT NULL REFERENCES workflow_runs (id) ON DELETE CASCADE,
    name        text NOT NULL,
    position    integer NOT NULL,
    status      text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'done', 'failed')),
    attempts    integer NOT NULL DEFAULT 0,
    output      jsonb,
    error       text,
    started_at  timestamptz,
    finished_at timestamptz,
    PRIMARY KEY (run_id, name)
);

-- Which sampled profile a respondent came from, so a resumed fan-out knows which ones are already done.
ALTER TABLE respondents ADD COLUMN IF NOT EXISTS slot integer;
CREATE UNIQUE INDEX IF NOT EXISTS respondents_slot_idx ON respondents (study_id, slot) WHERE slot IS NOT NULL;

-- M13: what happened during a workflow run, in order. Clients stream it and resume from the last id they saw.
CREATE TABLE IF NOT EXISTS run_events (
    id         bigserial PRIMARY KEY,
    run_id     bigint NOT NULL REFERENCES workflow_runs (id) ON DELETE CASCADE,
    type       text NOT NULL,
    data       jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS run_events_run_idx ON run_events (run_id, id);

-- M14: tie model calls to the workflow run that caused them (for "cost so far"), and add a human review gate.
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS run_id bigint;  -- no foreign key: the ledger must never block a run
CREATE INDEX IF NOT EXISTS llm_calls_run_idx ON llm_calls (run_id) WHERE run_id IS NOT NULL;
ALTER TABLE workflow_runs ADD COLUMN IF NOT EXISTS review_guide boolean NOT NULL DEFAULT false;
ALTER TABLE workflow_runs ADD COLUMN IF NOT EXISTS guide_approved boolean NOT NULL DEFAULT false;
