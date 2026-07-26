"""Build deterministic RAG knowledge-base chunks from structured tourism facts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

BASE_DIR = Path(__file__).resolve().parents[3]
FACTS_PATH = BASE_DIR / "data" / "raw" / "facts.jsonl"
SOURCES_PATH = BASE_DIR / "data" / "sources.csv"
DESTINATIONS_PATH = BASE_DIR / "data" / "destination_registry.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "knowledge_base.jsonl"
STATS_PATH = BASE_DIR / "artifacts" / "reports" / "knowledge_base_stats.json"

MIN_CHUNK_WORDS = 80
MAX_CHUNK_WORDS = 220
MIN_CHUNKS_PER_DESTINATION = 2
MIN_TOTAL_CHUNKS = 250
MAX_TOTAL_CHUNKS = 350
FOUR_CHUNK_FACT_THRESHOLD = 24

CATEGORY_ORDER = {
    "description": 0,
    "history": 1,
    "attraction": 2,
    "nature": 3,
    "activity": 4,
    "food": 5,
    "practical": 6,
    "route": 7,
}

CATEGORY_LABELS = {
    "description": "opis",
    "history": "istorija",
    "attraction": "atrakcije",
    "nature": "priroda",
    "activity": "aktivnosti",
    "food": "lokalna ponuda",
    "practical": "praktične preporuke",
    "route": "izleti i povezane destinacije",
}

SEMANTIC_GROUP = {
    "description": 0,
    "history": 0,
    "attraction": 0,
    "nature": 1,
    "activity": 1,
    "food": 2,
    "practical": 2,
    "route": 3,
}

REQUIRED_FACT_FIELDS = {
    "fact_id",
    "destination_id",
    "category",
    "text",
    "source_id",
    "last_verified_at",
}
REQUIRED_SOURCE_FIELDS = {"source_id", "url", "status"}
REQUIRED_DESTINATION_FIELDS = {"destination_id", "name"}


@dataclass(frozen=True)
class ChunkCandidate:
    start: int
    end: int
    categories: tuple[str, ...]
    text: str
    word_count: int
    cost: float


def count_words(text: str) -> int:
    return len(text.split())


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} in {path} must be a JSON object.")
            rows.append(row)
    return rows


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path} has no header.")
        rows = [dict(row) for row in reader]
    return list(reader.fieldnames), rows


def require_columns(path: Path, columns: Iterable[str], required: set[str]) -> None:
    missing = required - set(columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}.")


def load_destinations(path: Path) -> dict[str, str]:
    columns, rows = read_csv_rows(path)
    require_columns(path, columns, REQUIRED_DESTINATION_FIELDS)
    result: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        destination_id = row["destination_id"].strip()
        name = row["name"].strip()
        if not destination_id or not name:
            raise ValueError(f"Blank destination_id or name on row {row_number} in {path}.")
        if destination_id in result:
            raise ValueError(f"Duplicate destination_id {destination_id} in {path}.")
        result[destination_id] = name
    return result


def load_approved_source_urls(path: Path) -> dict[str, str]:
    columns, rows = read_csv_rows(path)
    require_columns(path, columns, REQUIRED_SOURCE_FIELDS)
    result: dict[str, str] = {}
    for row_number, row in enumerate(rows, start=2):
        source_id = row["source_id"].strip()
        url = row["url"].strip()
        status = row["status"].strip().lower()
        if status != "approved":
            continue
        if not source_id or not url:
            raise ValueError(f"Approved source on row {row_number} has blank ID or URL.")
        if source_id in result and result[source_id] != url:
            raise ValueError(f"Source ID {source_id} maps to multiple URLs in {path}.")
        result[source_id] = url
    if not result:
        raise ValueError(f"No approved sources found in {path}.")
    return result


def validate_and_prepare_facts(
    facts: Sequence[dict[str, Any]],
    destinations: dict[str, str],
    source_urls: dict[str, str],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    seen_fact_ids: set[str] = set()

    for position, raw_fact in enumerate(facts, start=1):
        missing = REQUIRED_FACT_FIELDS - set(raw_fact)
        if missing:
            raise ValueError(
                f"Fact at position {position} is missing: {', '.join(sorted(missing))}."
            )

        fact = dict(raw_fact)
        fact_id = str(fact["fact_id"]).strip()
        destination_id = str(fact["destination_id"]).strip()
        category = str(fact["category"]).strip()
        source_id = str(fact["source_id"]).strip()
        text = normalize_text(str(fact["text"]))
        last_verified_at = str(fact["last_verified_at"]).strip()

        if fact_id in seen_fact_ids:
            raise ValueError(f"Duplicate fact_id: {fact_id}.")
        seen_fact_ids.add(fact_id)
        if destination_id not in destinations:
            raise ValueError(f"Unknown destination_id {destination_id} in fact {fact_id}.")
        if category not in CATEGORY_ORDER:
            raise ValueError(f"Unsupported category {category} in fact {fact_id}.")
        if source_id not in source_urls:
            raise ValueError(f"Fact {fact_id} references missing or unapproved source {source_id}.")
        if not text:
            raise ValueError(f"Fact {fact_id} has empty text.")
        if not last_verified_at:
            raise ValueError(f"Fact {fact_id} has empty last_verified_at.")

        fact.update(
            fact_id=fact_id,
            destination_id=destination_id,
            category=category,
            source_id=source_id,
            text=text,
            last_verified_at=last_verified_at,
        )
        prepared.append(fact)

    return prepared


def ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def format_category_label(categories: Sequence[str]) -> str:
    category_set = set(categories)
    if category_set <= {"description", "history", "attraction"}:
        return "opis, istorija i atrakcije"
    if category_set <= {"nature", "activity"}:
        return "priroda i aktivnosti"
    if category_set <= {"food", "practical"}:
        return "lokalna ponuda i praktične preporuke"
    if category_set == {"route"}:
        return "izleti i povezane destinacije"

    labels = [CATEGORY_LABELS[category] for category in categories]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} i {labels[1]}"
    return ", ".join(labels[:-1]) + f" i {labels[-1]}"


def build_chunk_text(destination_name: str, facts: Sequence[dict[str, Any]]) -> tuple[str, tuple[str, ...]]:
    categories = tuple(
        sorted({fact["category"] for fact in facts}, key=CATEGORY_ORDER.__getitem__)
    )
    heading = f"{destination_name} — {format_category_label(categories)}."
    body = " ".join(fact["text"] for fact in facts)
    return f"{heading} {body}", categories


def candidate_cost(
    word_count: int,
    ideal_words: float,
    categories: Sequence[str],
) -> float:
    semantic_groups = {SEMANTIC_GROUP[category] for category in categories}
    mixing_penalty = 14.0 * max(0, len(semantic_groups) - 1)
    category_penalty = 1.5 * max(0, len(categories) - 2)
    return (word_count - ideal_words) ** 2 + mixing_penalty + category_penalty


def partition_facts(
    destination_name: str,
    facts: Sequence[dict[str, Any]],
    target_chunks: int,
) -> list[list[dict[str, Any]]]:
    if target_chunks < 1:
        raise ValueError("target_chunks must be positive.")
    if len(facts) < target_chunks:
        raise ValueError(
            f"Cannot create {target_chunks} non-empty chunks from {len(facts)} facts."
        )

    ordered_facts = sorted(
        facts,
        key=lambda fact: (
            SEMANTIC_GROUP[fact["category"]],
            CATEGORY_ORDER[fact["category"]],
            fact["fact_id"],
        ),
    )
    n = len(ordered_facts)
    total_fact_words = sum(count_words(fact["text"]) for fact in ordered_facts)
    ideal_words = (total_fact_words + target_chunks * 5) / target_chunks

    candidates: dict[tuple[int, int], ChunkCandidate] = {}
    for start in range(n):
        for end in range(start + 1, n + 1):
            segment = ordered_facts[start:end]
            text, categories = build_chunk_text(destination_name, segment)
            word_count = count_words(text)
            if word_count > MAX_CHUNK_WORDS:
                break
            if word_count < MIN_CHUNK_WORDS:
                continue
            candidates[(start, end)] = ChunkCandidate(
                start=start,
                end=end,
                categories=categories,
                text=text,
                word_count=word_count,
                cost=candidate_cost(word_count, ideal_words, categories),
            )

    infinity = math.inf
    dp = [[infinity] * (n + 1) for _ in range(target_chunks + 1)]
    previous: list[list[int | None]] = [
        [None] * (n + 1) for _ in range(target_chunks + 1)
    ]
    dp[0][0] = 0.0

    for chunk_number in range(1, target_chunks + 1):
        min_end = chunk_number
        for end in range(min_end, n + 1):
            for start in range(chunk_number - 1, end):
                candidate = candidates.get((start, end))
                if candidate is None or dp[chunk_number - 1][start] == infinity:
                    continue
                score = dp[chunk_number - 1][start] + candidate.cost
                if score < dp[chunk_number][end]:
                    dp[chunk_number][end] = score
                    previous[chunk_number][end] = start

    if dp[target_chunks][n] == infinity:
        raise ValueError(
            f"Could not split {destination_name} into {target_chunks} chunks "
            f"within {MIN_CHUNK_WORDS}-{MAX_CHUNK_WORDS} words."
        )

    boundaries: list[tuple[int, int]] = []
    end = n
    for chunk_number in range(target_chunks, 0, -1):
        start = previous[chunk_number][end]
        if start is None:
            raise RuntimeError("Internal partition reconstruction error.")
        boundaries.append((start, end))
        end = start
    boundaries.reverse()

    return [ordered_facts[start:end] for start, end in boundaries]


def target_chunk_count(fact_count: int) -> int:
    return 4 if fact_count >= FOUR_CHUNK_FACT_THRESHOLD else 3


def chunk_id_for(destination_id: str, number: int) -> str:
    slug = destination_id.replace("_", "-").upper()
    return f"CH-{slug}-{number:03d}"


def build_knowledge_base(
    facts: Sequence[dict[str, Any]],
    destinations: dict[str, str],
    source_urls: dict[str, str],
) -> list[dict[str, Any]]:
    facts_by_destination: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        facts_by_destination[fact["destination_id"]].append(fact)

    missing_destinations = [
        destination_id
        for destination_id in destinations
        if destination_id not in facts_by_destination
    ]
    if missing_destinations:
        raise ValueError(
            "Destinations without facts: " + ", ".join(missing_destinations)
        )

    chunks: list[dict[str, Any]] = []
    for destination_id, destination_name in destinations.items():
        destination_facts = facts_by_destination[destination_id]
        partitions = partition_facts(
            destination_name,
            destination_facts,
            target_chunk_count(len(destination_facts)),
        )

        for number, chunk_facts in enumerate(partitions, start=1):
            text, categories = build_chunk_text(destination_name, chunk_facts)
            source_ids = ordered_unique(fact["source_id"] for fact in chunk_facts)
            chunk = {
                "chunk_id": chunk_id_for(destination_id, number),
                "destination_id": destination_id,
                "destination_name": destination_name,
                "categories": list(categories),
                "text": text,
                "source_ids": source_ids,
                "source_urls": [source_urls[source_id] for source_id in source_ids],
                "last_verified_at": min(
                    fact["last_verified_at"] for fact in chunk_facts
                ),
            }
            chunks.append(chunk)

    return chunks


def validate_chunks(
    chunks: Sequence[dict[str, Any]],
    destinations: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    counts = Counter(chunk.get("destination_id") for chunk in chunks)
    seen_chunk_ids: set[str] = set()

    if not MIN_TOTAL_CHUNKS <= len(chunks) <= MAX_TOTAL_CHUNKS:
        errors.append(
            f"Expected {MIN_TOTAL_CHUNKS}-{MAX_TOTAL_CHUNKS} chunks, got {len(chunks)}."
        )

    for destination_id in destinations:
        if counts[destination_id] < MIN_CHUNKS_PER_DESTINATION:
            errors.append(
                f"{destination_id} has {counts[destination_id]} chunks; "
                f"minimum is {MIN_CHUNKS_PER_DESTINATION}."
            )

    required_fields = {
        "chunk_id",
        "destination_id",
        "destination_name",
        "categories",
        "text",
        "source_ids",
        "source_urls",
        "last_verified_at",
    }
    for position, chunk in enumerate(chunks, start=1):
        missing = required_fields - set(chunk)
        if missing:
            errors.append(
                f"Chunk at position {position} is missing: {', '.join(sorted(missing))}."
            )
            continue
        chunk_id = chunk["chunk_id"]
        if chunk_id in seen_chunk_ids:
            errors.append(f"Duplicate chunk_id {chunk_id}.")
        seen_chunk_ids.add(chunk_id)

        word_count = count_words(chunk["text"])
        if not MIN_CHUNK_WORDS <= word_count <= MAX_CHUNK_WORDS:
            errors.append(
                f"{chunk_id} has {word_count} words; allowed range is "
                f"{MIN_CHUNK_WORDS}-{MAX_CHUNK_WORDS}."
            )
        if not chunk["text"].strip():
            errors.append(f"{chunk_id} has empty text.")
        if not chunk["source_ids"] or not chunk["source_urls"]:
            errors.append(f"{chunk_id} has no sources.")
        if len(chunk["source_ids"]) != len(chunk["source_urls"]):
            errors.append(f"{chunk_id} has mismatched source IDs and URLs.")
        if not chunk["categories"]:
            errors.append(f"{chunk_id} has no categories.")

    return errors


def build_stats(
    chunks: Sequence[dict[str, Any]],
    destinations: dict[str, str],
    errors: Sequence[str],
) -> dict[str, Any]:
    counts = Counter(chunk["destination_id"] for chunk in chunks)
    category_counts = Counter(
        category for chunk in chunks for category in chunk["categories"]
    )
    word_counts = [count_words(chunk["text"]) for chunk in chunks]
    source_ids = {
        source_id for chunk in chunks for source_id in chunk["source_ids"]
    }

    return {
        "success": not errors,
        "chunk_count": len(chunks),
        "destination_count": len(counts),
        "source_count": len(source_ids),
        "word_count": {
            "minimum": min(word_counts) if word_counts else 0,
            "maximum": max(word_counts) if word_counts else 0,
            "average": round(statistics.mean(word_counts), 2) if word_counts else 0,
            "median": round(statistics.median(word_counts), 2) if word_counts else 0,
        },
        "chunks_per_destination": {
            destination_id: counts[destination_id]
            for destination_id in destinations
        },
        "category_chunk_counts": dict(sorted(category_counts.items())),
        "critical_errors": list(errors),
    }


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, default=FACTS_PATH)
    parser.add_argument("--sources", type=Path, default=SOURCES_PATH)
    parser.add_argument("--destinations", type=Path, default=DESTINATIONS_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--stats-out", type=Path, default=STATS_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    destinations = load_destinations(args.destinations)
    source_urls = load_approved_source_urls(args.sources)
    raw_facts = read_jsonl(args.facts)
    facts = validate_and_prepare_facts(raw_facts, destinations, source_urls)
    chunks = build_knowledge_base(facts, destinations, source_urls)
    errors = validate_chunks(chunks, destinations)
    stats = build_stats(chunks, destinations, errors)

    write_json(args.stats_out, stats)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    write_jsonl(args.out, chunks)
    print(
        f"Knowledge base built successfully: {len(chunks)} chunks for "
        f"{len(destinations)} destinations."
    )
    print(f"Output: {args.out}")
    print(f"Stats: {args.stats_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
