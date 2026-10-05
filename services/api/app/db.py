import hashlib
import json
import logging
import os
from contextvars import ContextVar

import asyncpg

from app.prompts.registry import PromptRef

logger = logging.getLogger(__name__)

# The workflow run whose work is going on right now. Every model call made while it is set is logged with the run's id.
# A ContextVar follows the task and the tasks it starts, so parallel analysts are covered too.
current_run: ContextVar[int | None] = ContextVar("current_run", default=None)

DEFAULT_URL = "postgresql://insightloop:insightloop@localhost:5433/insightloop"


async def connect() -> asyncpg.Connection:
    return await asyncpg.connect(os.environ.get("DATABASE_URL", DEFAULT_URL), timeout=3)


async def fetch_call(call_id: int) -> asyncpg.Record | None:
    conn = await connect()
    try:
        return await conn.fetchrow("SELECT * FROM llm_calls WHERE id = $1", call_id)
    finally:
        await conn.close()


def inputs_hash(inputs: dict) -> str:
    """Fingerprint of a call's inputs. The same inputs always give the same hash."""
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


async def record_call(
    *,
    prompt: PromptRef | None,
    model: str,
    messages: list[dict[str, str]],
    params: dict,
    output: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    cost_usd: float | None,
) -> None:
    """Write one row to llm_calls. A ledger failure is logged, never raised."""
    try:
        conn = await connect()
        try:
            await conn.execute(
                """
                INSERT INTO llm_calls (prompt_name, prompt_version, inputs, inputs_hash,
                                       model, messages, params, output,
                                       input_tokens, output_tokens, latency_ms, cost_usd, run_id)
                VALUES ($1, $2, $3::jsonb, $4, $5, $6::jsonb, $7::jsonb, $8, $9, $10, $11, $12, $13)
                """,
                prompt.name if prompt else None,
                prompt.version if prompt else None,
                json.dumps(prompt.inputs) if prompt else None,
                inputs_hash(prompt.inputs) if prompt else None,
                model,
                json.dumps(messages),
                json.dumps(params),
                output,
                input_tokens,
                output_tokens,
                latency_ms,
                cost_usd,
                current_run.get(),
            )
        finally:
            await conn.close()
    except Exception:
        logger.exception("could not record llm call in llm_calls")
