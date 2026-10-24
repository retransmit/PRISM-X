"""Write the live loop's stream with the shipped generator.

    python -m prism.live.generate

Four months at the ``live:`` settings, on a seed neither the training export nor the
harness uses, so the stream the runner replays is one no model has seen.
"""

from __future__ import annotations

import subprocess
import sys

from prism import config
from prism.harness.data import generator_command


def command(s: dict) -> list[str]:
    return generator_command(s, int(s["seed"]), config.get_path("live.stem"))


def main() -> None:
    s = dict(config.get("live"))
    stem = config.get_path("live.stem")
    stem.parent.mkdir(parents=True, exist_ok=True)
    chosen = command(s)
    print(f"[live] {' '.join(chosen)}", file=sys.stderr)
    subprocess.run(chosen, check=True)


if __name__ == "__main__":
    main()
