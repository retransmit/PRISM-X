"""Write the harness's seed streams with the shipped generator.

    python -m prism.harness.generate [--seeds 42,7]
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from prism.harness import data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the harness's seed streams with the shipped generator.")
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated seeds; default: the config's")
    args = parser.parse_args()
    chosen = [int(s) for s in args.seeds.split(",")] if args.seeds else data.seeds()
    for seed in chosen:
        data.stem(seed).parent.mkdir(parents=True, exist_ok=True)
        command = data.generate_command(seed)
        print(f"[harness] seed {seed}: {' '.join(command)}", file=sys.stderr)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
