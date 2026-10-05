"""Records the hash of every prompt file so a published version can never change silently.

Run `python -m app.prompts.manifest` (or `make lock-prompts`) after adding a new version.
"""

import hashlib
import json
from pathlib import Path

from app.prompts.registry import PROMPTS_DIR

MANIFEST_PATH = PROMPTS_DIR / "manifest.json"


def current_hashes(prompts_dir: Path = PROMPTS_DIR) -> dict[str, str]:
    return {
        str(path.relative_to(prompts_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(prompts_dir.glob("*/v*.md"))
    }


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, str]:
    return json.loads(path.read_text()) if path.exists() else {}


def problems(recorded: dict[str, str], current: dict[str, str]) -> list[str]:
    found = []
    for name, digest in current.items():
        if name not in recorded:
            found.append(f"{name} is not in the manifest. Run `make lock-prompts` to record it.")
        elif recorded[name] != digest:
            found.append(
                f"{name} changed but its version did not. "
                "Restore it and put your edit in a new version file."
            )
    for name in recorded:
        if name not in current:
            found.append(f"{name} was deleted. Old versions must stay so past calls can be replayed.")
    return found


def lock() -> None:
    """Record new prompt files. Never rewrites an existing entry."""
    recorded, current = load_manifest(), current_hashes()
    broken = sorted(name for name, digest in recorded.items() if current.get(name) != digest)
    if broken:
        raise SystemExit(f"Refusing to lock: {', '.join(broken)} changed or was deleted.")
    MANIFEST_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    added = sorted(set(current) - set(recorded))
    print(f"recorded {len(added)} new prompt file(s): {', '.join(added) or 'none'}")


if __name__ == "__main__":
    lock()
