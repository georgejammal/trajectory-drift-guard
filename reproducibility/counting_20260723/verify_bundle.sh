#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE="$ROOT/reproducibility/counting_20260723"

python - "$ROOT" "$BUNDLE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
bundle = Path(sys.argv[2])
provenance = json.loads((bundle / "provenance.json").read_text())
failed = []
for relative_path, expected in provenance["frozen_file_sha256"].items():
    path = root / relative_path
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    if actual != expected:
        failed.append((relative_path, expected, actual))

if failed:
    for path, expected, actual in failed:
        print(f"MISMATCH {path}: expected={expected} actual={actual}")
    raise SystemExit(1)

print(f"verified {len(provenance['frozen_file_sha256'])} frozen source files")
PY
