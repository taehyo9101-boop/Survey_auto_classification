from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv


REQUIRED_COLUMNS = {"idx", "content"}


def load_project_env(env_path: Path) -> None:
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found: {env_path}")
    load_dotenv(env_path, override=False)


def read_survey_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    sample = path.read_bytes()[:4096]
    encoding = "utf-8-sig" if sample.startswith(b"\xef\xbb\xbf") else "utf-8"
    df = pd.read_csv(path, encoding=encoding)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")
    return df


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def latest_run_dir(output_root: Path, phase_name: str, *, preferred_name: str | None = None) -> Path:
    phase_dir = output_root / phase_name
    if not phase_dir.exists():
        raise FileNotFoundError(f"No runs found for phase '{phase_name}' in {phase_dir}")
    if preferred_name:
        preferred_dir = phase_dir / preferred_name
        if preferred_dir.exists() and preferred_dir.is_dir():
            return preferred_dir
    candidates = sorted([path for path in phase_dir.iterdir() if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No runs found for phase '{phase_name}' in {phase_dir}")
    return candidates[-1]
