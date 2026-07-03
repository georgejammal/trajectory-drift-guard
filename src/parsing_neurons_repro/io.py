from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_records(path: Path, list_key: str | None = None) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if list_key is not None:
            rows = payload[list_key]
        elif "pairs" in payload:
            rows = payload["pairs"]
        elif "rows" in payload:
            rows = payload["rows"]
        else:
            raise ValueError(f"Could not infer record list key in {path}.")
        if not isinstance(rows, list):
            raise ValueError(f"Record field in {path} is not a list.")
        return rows
    raise ValueError(f"Unsupported record payload type in {path}: {type(payload).__name__}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_csv(raw: str | list[Any]) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def parse_float_grid(raw: str | list[Any]) -> list[float]:
    return [float(item) for item in parse_csv(raw)]


def parse_window(raw: str) -> tuple[int, int]:
    start, end = raw.split("-", 1)
    return int(start), int(end)


def window_layers(raw: str) -> list[int]:
    start, end = parse_window(raw)
    return list(range(start, end + 1))


def clean_float(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace(".", "p")


def slug_parts(*parts: Any) -> str:
    return "_".join(str(part).replace("/", "-").replace(" ", "-") for part in parts if part is not None)
