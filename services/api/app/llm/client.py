import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.db import record_call
from app.llm.pricing import estimate_cost, estimate_embedding_cost
from app.llm.retry import with_retries
from app.prompts.registry import PromptRef

load_dotenv()  # reads OPENAI_API_KEY from the repo's .env into the environment

logger = logging.getLogger(__name__)
_client = AsyncOpenAI(max_retries=0)  # our own with_retries handles retrying

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float | None


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]  # one per input text, in the same order
    input_tokens: int
    latency_ms: float
    cost_usd: float | None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # the JSON text exactly as the model wrote it, which may not be valid JSON


@dataclass
class ToolTurn:
    content: str | None  # the model's text, usually empty when it calls tools
    tool_calls: list[ToolCall]
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float | None

    def assistant_message(self) -> dict:
        """This turn in the shape the API expects back in the conversation history."""
        message: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return message


class LLMRefusal(Exception):
    """The model declined to answer instead of returning the requested structure."""


class LLMInvalidOutput(Exception):
    """The model's output still failed validation after the retry."""


def _log_call(model: str, usage, start: float) -> tuple[float, float | None]:
    latency_ms = (time.perf_counter() - start) * 1000
    cost_usd = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
    logger.info(
        "llm_call model=%s input_tokens=%d output_tokens=%d latency_ms=%.0f cost_usd=%s",
        model,
        usage.prompt_tokens,
        usage.completion_tokens,
        latency_ms,
        f"{cost_usd:.6f}" if cost_usd is not None else "unknown",
    )
    return latency_ms, cost_usd


async def complete(
    messages: list[dict[str, str]], model: str, prompt: PromptRef | None = None
) -> LLMResult:
    start = time.perf_counter()
    response = await with_retries(
        lambda: _client.chat.completions.create(model=model, messages=messages)
    )
    usage = response.usage
    latency_ms, cost_usd = _log_call(model, usage, start)
    text = response.choices[0].message.content or ""
    await record_call(
        prompt=prompt,
        model=model,
        messages=messages,
        params={},
        output=text,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )
    return LLMResult(
        text=text,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )


async def complete_streaming(
    messages: list[dict[str, str]], model: str, on_text: Callable[[str], Awaitable[None]], prompt: PromptRef | None = None
) -> LLMResult:
    """Like `complete`, but calls `on_text` with each piece of the reply as it arrives, and returns the whole reply.

    Retries cover only opening the stream. If the connection breaks half-way the error propagates, because the
    pieces already sent to `on_text` cannot be taken back (the caller decides what a restart means).
    """
    start = time.perf_counter()
    stream = await with_retries(
        lambda: _client.chat.completions.create(
            model=model, messages=messages, stream=True, stream_options={"include_usage": True}
        )
    )
    parts: list[str] = []
    usage = None
    try:
        async for chunk in stream:
            if getattr(chunk, "usage", None):  # the last chunk carries the token counts and has no choices
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
                await on_text(parts[-1])
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            await close()  # stop reading (and paying) if the callback raised
    if usage is None:
        raise RuntimeError("The stream ended without token counts, so the reply may be cut off.")
    latency_ms, cost_usd = _log_call(model, usage, start)
    text = "".join(parts)
    await record_call(
        prompt=prompt, model=model, messages=messages, params={"stream": True}, output=text,
        input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens, latency_ms=latency_ms, cost_usd=cost_usd,
    )
    return LLMResult(text, usage.prompt_tokens, usage.completion_tokens, latency_ms, cost_usd)


async def embed(texts: list[str], model: str) -> EmbeddingResult:
    """Turn each text into a vector with one API request. Not for empty lists or empty strings."""
    start = time.perf_counter()
    response = await with_retries(lambda: _client.embeddings.create(model=model, input=texts))
    latency_ms = (time.perf_counter() - start) * 1000
    tokens = response.usage.prompt_tokens
    cost_usd = estimate_embedding_cost(model, tokens)
    logger.info(
        "embed_call model=%s inputs=%d tokens=%d latency_ms=%.0f cost_usd=%s",
        model,
        len(texts),
        tokens,
        latency_ms,
        f"{cost_usd:.6f}" if cost_usd is not None else "unknown",
    )
    ordered = sorted(response.data, key=lambda item: item.index)  # do not trust the response order
    return EmbeddingResult([item.embedding for item in ordered], tokens, latency_ms, cost_usd)


async def complete_with_tools(
    messages: list[dict],
    model: str,
    tools: list[dict],
    tool_choice: str = "auto",
    prompt: PromptRef | None = None,
) -> ToolTurn:
    """One model turn in which it may ask for tools. tool_choice="none" forbids tool calls (forces an answer)."""
    start = time.perf_counter()
    response = await with_retries(
        lambda: _client.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice=tool_choice
        )
    )
    usage = response.usage
    latency_ms, cost_usd = _log_call(model, usage, start)
    message = response.choices[0].message
    calls = [ToolCall(c.id, c.function.name, c.function.arguments) for c in (message.tool_calls or [])]
    await record_call(
        prompt=prompt,
        model=model,
        messages=messages,
        params={"tools": [t["function"]["name"] for t in tools], "tool_choice": tool_choice},
        output=message.content or json.dumps([{"name": c.name, "arguments": c.arguments} for c in calls]),
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )
    return ToolTurn(message.content, calls, usage.prompt_tokens, usage.completion_tokens, latency_ms, cost_usd)


async def _parse_once(
    messages: list[dict[str, str]], model: str, schema: type[T], prompt: PromptRef | None
) -> T:
    start = time.perf_counter()
    response = await with_retries(
        lambda: _client.chat.completions.parse(
            model=model, messages=messages, response_format=schema
        )
    )
    usage = response.usage
    latency_ms, cost_usd = _log_call(model, usage, start)
    message = response.choices[0].message
    await record_call(
        prompt=prompt,
        model=model,
        messages=messages,
        params={"response_format": schema.__name__},
        output=message.refusal or message.parsed.model_dump_json(),
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )
    if message.refusal:
        raise LLMRefusal(message.refusal)
    return message.parsed


async def complete_structured(
    messages: list[dict[str, str]],
    model: str,
    schema: type[T],
    prompt: PromptRef | None = None,
) -> T:
    """Ask the model for output matching `schema`; retry once with the error fed back."""
    try:
        return await _parse_once(messages, model, schema, prompt)
    except ValidationError as error:
        logger.warning("structured output failed validation, retrying once: %s", error)
        feedback = (
            f"Your previous response failed validation:\n{error}\n"
            "Respond again with output that fixes these problems."
        )
        retry_messages = messages + [{"role": "user", "content": feedback}]
        try:
            return await _parse_once(retry_messages, model, schema, prompt)
        except ValidationError as second_error:
            raise LLMInvalidOutput(str(second_error)) from second_error
