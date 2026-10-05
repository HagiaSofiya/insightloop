import asyncio
import logging
import sys

from app.llm.client import complete

MODEL = "gpt-4o-mini"
DEFAULT_PROMPT = "In two sentences, why do consumer researchers ask open-ended questions?"


async def main(prompt: str) -> None:
    result = await complete([{"role": "user", "content": prompt}], MODEL)
    cost = f"${result.cost_usd:.6f}" if result.cost_usd is not None else "unknown"
    print(f"\n{result.text}\n")
    print(f"model:    {MODEL}")
    print(f"tokens:   {result.input_tokens} in, {result.output_tokens} out")
    print(f"latency:  {result.latency_ms:.0f} ms")
    print(f"cost:     {cost}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT))
