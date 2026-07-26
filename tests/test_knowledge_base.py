"""Tests for the deterministic T10 RAG knowledge-base builder."""

import csv
import json
from collections import Counter
from pathlib import Path

from src.bih_guide.data.build_knowledge_base import (
    MAX_CHUNK_WORDS,
    MIN_CHUNK_WORDS,
    build_chunk_text,
    build_knowledge_base,
    count_words,
    partition_facts,
    target_chunk_count,
)

BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "processed" / "knowledge_base.jsonl"
FACTS_PATH = BASE_DIR / "data" / "raw" / "facts.jsonl"
SOURCES_PATH = BASE_DIR / "data" / "sources.csv"
DESTINATIONS_PATH = BASE_DIR / "data" / "destination_registry.csv"
STATS_PATH = BASE_DIR / "artifacts" / "reports" / "knowledge_base_stats.json"

REQUIRED_FIELDS = {
    "chunk_id",
    "destination_id",
    "destination_name",
    "categories",
    "text",
    "source_ids",
    "source_urls",
    "last_verified_at",
}


def make_fact(number: int, category: str = "history") -> dict:
    """Create a short valid fact suitable for partitioning tests."""
    return {
        "fact_id": f"F-TEST-{number:03d}",
        "destination_id": "test",
        "category": category,
        "text": (
            f"Ovo je provjerena turistička činjenica broj {number} koja sadrži "
            "dovoljno riječi za pouzdano testiranje segmentacije odlomaka."
        ),
        "source_id": "SRC-TEST-001",
        "is_dynamic": False,
        "last_verified_at": "2026-07-25",
    }


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_target_chunk_count_uses_three_or_four_chunks():
    assert target_chunk_count(23) == 3
    assert target_chunk_count(24) == 4


def test_partition_facts_respects_word_bounds():
    categories = ["description", "history", "attraction", "nature", "activity", "practical"]
    facts = [make_fact(i, categories[i % len(categories)]) for i in range(24)]

    partitions = partition_facts("Testna destinacija", facts, target_chunks=4)

    assert len(partitions) == 4
    assert sum(len(partition) for partition in partitions) == len(facts)
    for partition in partitions:
        text, _ = build_chunk_text("Testna destinacija", partition)
        assert MIN_CHUNK_WORDS <= count_words(text) <= MAX_CHUNK_WORDS


def test_build_knowledge_base_has_required_schema_and_urls():
    facts = [make_fact(i) for i in range(24)]
    destinations = {"test": "Testna destinacija"}
    source_urls = {"SRC-TEST-001": "https://example.org/test"}

    chunks = build_knowledge_base(facts, destinations, source_urls)
    assert len(chunks) == 4
    for chunk in chunks:
        assert REQUIRED_FIELDS <= set(chunk)
        assert chunk["source_ids"] == ["SRC-TEST-001"]
        assert chunk["source_urls"] == ["https://example.org/test"]
        assert chunk["text"].startswith("Testna destinacija —")
        assert MIN_CHUNK_WORDS <= count_words(chunk["text"]) <= MAX_CHUNK_WORDS


def test_generated_repository_knowledge_base_meets_t10_requirements():
    assert KNOWLEDGE_BASE_PATH.exists(), (
        "Pokrenite python -m bih_guide.data.build_knowledge_base prije testova."
    )
    chunks = read_jsonl(KNOWLEDGE_BASE_PATH)

    assert 250 <= len(chunks) <= 350
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)

    destination_counts = Counter(chunk["destination_id"] for chunk in chunks)
    with DESTINATIONS_PATH.open(encoding="utf-8-sig", newline="") as handle:
        destinations = list(csv.DictReader(handle))

    assert len(destinations) == 72
    assert set(destination_counts) == {row["destination_id"] for row in destinations}
    assert min(destination_counts.values()) >= 2

    for chunk in chunks:
        assert REQUIRED_FIELDS <= set(chunk)
        assert chunk["text"].strip()
        assert MIN_CHUNK_WORDS <= count_words(chunk["text"]) <= MAX_CHUNK_WORDS
        assert chunk["categories"]
        assert chunk["source_ids"]
        assert chunk["source_urls"]
        assert len(chunk["source_ids"]) == len(chunk["source_urls"])
        assert chunk["text"].startswith(f"{chunk['destination_name']} —")


def test_every_source_url_exists_in_approved_source_registry():
    chunks = read_jsonl(KNOWLEDGE_BASE_PATH)
    with SOURCES_PATH.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    approved = {
        row["source_id"]: row["url"]
        for row in source_rows
        if row["status"].strip().lower() == "approved"
    }

    for chunk in chunks:
        assert chunk["source_urls"] == [
            approved[source_id] for source_id in chunk["source_ids"]
        ]


def test_every_fact_text_is_used_once_inside_its_destination_chunks():
    facts = read_jsonl(FACTS_PATH)
    chunks = read_jsonl(KNOWLEDGE_BASE_PATH)
    chunks_by_destination: dict[str, list[str]] = {}
    for chunk in chunks:
        chunks_by_destination.setdefault(chunk["destination_id"], []).append(chunk["text"])

    for fact in facts:
        occurrences = sum(
            fact["text"] in chunk_text
            for chunk_text in chunks_by_destination[fact["destination_id"]]
        )
        assert occurrences == 1, fact["fact_id"]


def test_stats_report_matches_generated_output():
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    chunks = read_jsonl(KNOWLEDGE_BASE_PATH)

    assert stats["success"] is True
    assert stats["critical_errors"] == []
    assert stats["chunk_count"] == len(chunks)
    assert stats["destination_count"] == 72
    assert stats["word_count"]["minimum"] >= MIN_CHUNK_WORDS
    assert stats["word_count"]["maximum"] <= MAX_CHUNK_WORDS
