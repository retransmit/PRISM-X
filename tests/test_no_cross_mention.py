"""Nothing in this repository names the generator's own development line or its author.

The generator's own calibration report is copied beside the metrics by the
training run, and the generator itself is mirrored under generator/ by a tool that
rewrites its names; both scans keep it that way. The names are held encoded, so the
scan covers every file, this one included."""

from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path

from prism import config

NAMES = re.compile(base64.b64decode("ZnJhdWR4fGJoYXZ5").decode(), re.IGNORECASE)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".ts", ".tsx", ".css", ".html", ".csv", ".sh", ".ps1", ""}


def _offenders(paths: list[Path]) -> list[str]:
    found = []
    for path in paths:
        rel = path.relative_to(config.PROJECT_ROOT).as_posix()
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if NAMES.search(path.read_text(encoding="utf-8", errors="ignore")):
            found.append(rel)
    return found


def test_the_committed_metrics_never_name_the_generator():
    root = config.PROJECT_ROOT / "reports" / "metrics"
    assert _offenders([path for path in root.rglob("*") if path.suffix in {".json", ".md"}]) == []


def test_no_tracked_file_names_the_generators_development_line():
    listing = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=config.PROJECT_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split("\n")
    paths = [config.PROJECT_ROOT / line.strip() for line in listing if line.strip()]
    assert _offenders(paths) == []
