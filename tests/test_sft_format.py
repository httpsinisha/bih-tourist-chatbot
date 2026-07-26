"""Tests for deterministic T11 SFT candidate generation and output format."""

import csv
import json
from collections import Counter
from pathlib import Path

from src.bih_guide.data.generate_sft import (
    DESTINATION_TYPES,
    NON_FACTUAL_TYPES,
    generate_all_candidates,
    load_templates,
    read_destinations,
    read_jsonl,
    validate_candidates,
)

BASE_DIR = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "processed" / "knowledge_base.jsonl"
DESTINATIONS_PATH = BASE_DIR / "data" / "destination_registry.csv"
TEMPLATES_PATH = BASE_DIR / "configs" / "sft_templates.yaml"
CANDIDATES_PATH = BASE_DIR / "data" / "processed" / "sft_candidates.jsonl"
STATS_PATH = BASE_DIR / "artifacts" / "reports" / "sft_candidate_stats.json"

REQUIRED_FIELDS = {
    "example_id",
    "type",
    "destination_ids",
    "chunk_ids",
    "source_ids",
    "reviewed",
    "review_status",
    "question_family",
    "messages",
}


def repository_candidates() -> list[dict]:
    assert CANDIDATES_PATH.exists(), (
        "Pokrenite python -m bih_guide.data.generate_sft prije testova."
    )
    return read_jsonl(CANDIDATES_PATH)


def test_templates_define_all_eight_destination_types():
    config = load_templates(TEMPLATES_PATH)
    configured_types = {
        template["type"]
        for template in config["destination_templates"]
    }

    assert DESTINATION_TYPES <= configured_types
    assert len(config["destination_templates"]) >= 8


def test_generated_candidates_meet_t11_total_and_type_counts():
    examples = repository_candidates()
    type_counts = Counter(example["type"] for example in examples)

    assert len(examples) >= 750
    assert type_counts["multi_destination_plan"] >= 100
    assert type_counts["clarification"] >= 50
    assert type_counts["multi_turn"] >= 60
    assert type_counts["uncertain_dynamic"] >= 40
    assert type_counts["out_of_domain"] >= 30


def test_every_destination_has_at_least_eight_base_candidates():
    examples = repository_candidates()
    destinations = read_destinations(DESTINATIONS_PATH)
    counts = Counter(
        example["destination_ids"][0]
        for example in examples
        if example["type"] in DESTINATION_TYPES
        and len(example["destination_ids"]) == 1
    )

    assert len(destinations) == 72
    for destination in destinations:
        assert counts[destination["destination_id"]] >= 8


def test_candidate_schema_and_message_roles_are_valid():
    for example in repository_candidates():
        assert REQUIRED_FIELDS <= set(example)
        assert example["reviewed"] is False
        assert example["review_status"] == "pending"
        assert example["messages"][0]["role"] == "system"
        assert example["messages"][-1]["role"] == "assistant"
        assert len(example["messages"]) >= 3

        for message in example["messages"]:
            assert message["role"] in {"system", "user", "assistant"}
            assert isinstance(message["content"], str)
            assert message["content"].strip()
            assert "{" not in message["content"]
            assert "}" not in message["content"]


def test_example_ids_and_user_messages_are_unique():
    examples = repository_candidates()
    example_ids = [example["example_id"] for example in examples]
    user_messages = [
        " ".join(message["content"].split()).casefold()
        for example in examples
        for message in example["messages"]
        if message["role"] == "user"
    ]

    assert len(example_ids) == len(set(example_ids))
    assert len(user_messages) == len(set(user_messages))


def test_factual_candidates_have_known_source_ids():
    examples = repository_candidates()
    chunks = read_jsonl(KNOWLEDGE_BASE_PATH)
    known_source_ids = {
        source_id
        for chunk in chunks
        for source_id in chunk["source_ids"]
    }

    for example in examples:
        assert set(example["source_ids"]) <= known_source_ids
        if example["type"] not in NON_FACTUAL_TYPES:
            assert example["source_ids"]


def test_multi_turn_examples_have_expected_role_sequence():
    examples = repository_candidates()
    multi_turn = [
        example
        for example in examples
        if example["type"] == "multi_turn"
    ]

    assert len(multi_turn) >= 60
    for example in multi_turn:
        assert [message["role"] for message in example["messages"]] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
        ]


def test_generator_is_deterministic_and_validator_accepts_output():
    chunks = read_jsonl(KNOWLEDGE_BASE_PATH)
    destinations = read_destinations(DESTINATIONS_PATH)
    config = load_templates(TEMPLATES_PATH)

    first = generate_all_candidates(chunks, destinations, config)
    second = generate_all_candidates(chunks, destinations, config)
    assert first == second

    known_source_ids = {
        source_id
        for chunk in chunks
        for source_id in chunk["source_ids"]
    }
    assert validate_candidates(
        first,
        destinations,
        known_source_ids,
        minimum_total=int(config["minimum_total_candidates"]),
    ) == []


def test_stats_report_matches_candidate_file():
    examples = repository_candidates()
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    type_counts = Counter(example["type"] for example in examples)

    assert stats["success"] is True
    assert stats["critical_errors"] == []
    assert stats["candidate_count"] == len(examples)
    assert stats["type_counts"] == dict(sorted(type_counts.items()))
    assert stats["unique_user_message_count"] == stats["user_message_count"]
    assert min(stats["base_templates_per_destination"].values()) >= 8
