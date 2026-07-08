#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


LANGUAGE_SPECS = {
    "Arabic": {"code": "ar", "pair": "ar-en", "english_file": "News-Commentary.ar-en.en", "target_file": "News-Commentary.ar-en.ar"},
    "Japanese": {"code": "ja", "pair": "en-ja", "english_file": "News-Commentary.en-ja.en", "target_file": "News-Commentary.en-ja.ja"},
    "Russian": {"code": "ru", "pair": "en-ru", "english_file": "News-Commentary.en-ru.en", "target_file": "News-Commentary.en-ru.ru"},
}


def read_lines(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def download_and_extract(raw_root: Path, pair: str, *, force: bool = False) -> None:
    raw_root.mkdir(parents=True, exist_ok=True)
    zip_path = raw_root / f"{pair}.txt.zip"
    if force or not zip_path.exists():
        url = f"https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/{pair}.txt.zip"
        print(f"[news-commentary] downloading {url}", flush=True)
        urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_root)


def materialize_language(raw_root: Path, output_root: Path, language: str, *, max_pairs: int) -> dict:
    spec = LANGUAGE_SPECS[language]
    english = read_lines(raw_root / spec["english_file"])
    target = read_lines(raw_root / spec["target_file"])
    paired = [(en.strip(), tgt.strip()) for en, tgt in zip(english, target)]
    total_pairs = len(paired)
    paired = paired[:max_pairs]
    code = spec["code"]
    out_dir = output_root / f"en-{code}"
    write_lines(out_dir / "train.en", [en for en, _ in paired])
    write_lines(out_dir / f"train.{code}", [tgt for _, tgt in paired])
    return {
        "language": language,
        "code": code,
        "pair": spec["pair"],
        "num_source_pairs": total_pairs,
        "num_pairs": len(paired),
        "selection": f"first_{max_pairs}_aligned_pairs",
        "english_path": str((out_dir / "train.en").resolve()),
        "target_path": str((out_dir / f"train.{code}").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OPUS News-Commentary v16 and materialize the data/ncwm layout expected by INCLINE."
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/news_commentary_v16"))
    parser.add_argument("--output-root", type=Path, default=Path("data/ncwm"))
    parser.add_argument("--languages", default="Arabic,Japanese,Russian")
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=500,
        help="Write only the first N aligned pairs. The released INCLINE scripts only iterate over ind < 500.",
    )
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    unsupported = sorted(set(languages) - set(LANGUAGE_SPECS))
    if unsupported:
        raise ValueError(
            "News Commentary v16 does not provide these requested language pairs "
            f"in the INCLINE English-target setup: {unsupported}. "
            f"Available here: {sorted(LANGUAGE_SPECS)}"
        )

    records = []
    for language in languages:
        download_and_extract(args.raw_root, LANGUAGE_SPECS[language]["pair"], force=args.force_download)
        records.append(materialize_language(args.raw_root, args.output_root, language, max_pairs=args.max_pairs))

    metadata = {
        "source": "OPUS News-Commentary v16 Moses downloads",
        "source_url_template": "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/moses/{pair}.txt.zip",
        "note": "This materializes the data/ncwm/en-{lang}/train.en and train.{lang} layout used by the released INCLINE scripts. By default it writes only the first 500 aligned pairs because the released scripts only iterate over ind < 500.",
        "max_pairs": args.max_pairs,
        "languages": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
