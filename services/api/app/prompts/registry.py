from dataclasses import dataclass
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).parent


@dataclass
class PromptRef:
    """Which prompt a call used and the user inputs that went into it."""

    name: str
    version: str
    inputs: dict


def load(name: str, version: str, /, **variables: str) -> str:
    """Return prompts/<name>/<version>.md with $variables filled in.

    Raises FileNotFoundError for an unknown prompt and KeyError for a missing variable.
    """
    text = (PROMPTS_DIR / name / f"{version}.md").read_text()
    return Template(text).substitute(variables)


def build_messages(name: str, version: str, inputs: dict) -> list[dict[str, str]]:
    """The prompt file is the system message and the input brief is the user message."""
    return [
        {"role": "system", "content": load(name, version, **inputs)},
        {"role": "user", "content": inputs["brief"]},
    ]
