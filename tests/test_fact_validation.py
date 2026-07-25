"""
Tests for src/bih_guide/data/validate_facts.py
"""

import json
from datetime import date, timedelta

import pytest

from src.bih_guide.data.validate_facts import (
    MIN_FACT_PER_DESTINATION,
    MIN_TEXT_LENGTH,
    MAX_TEXT_LENGTH,
    read_jsonl,
    create_destination_fact_map,
    validate_required_fields,
    normalize_text,
    is_text_invalid,
    validate_destination_id,
    validate_source_id,
    deduplicate_destination_facts,
    validate_destination_facts,
    validate_facts,
)


def make_fact(
    fact_id="f1",
    destination_id="sarajevo",
    category="history",
    text="A" * 50,
    source_id="src1",
    is_dynamic=False,
    last_verified_at="2026-01-01",
):
    """Build a valid fact dict, overriding individual fields as needed."""
    fact = {
        "fact_id": fact_id,
        "destination_id": destination_id,
        "category": category,
        "text": text,
        "source_id": source_id,
        "is_dynamic": is_dynamic,
    }
    if last_verified_at is not None:
        fact["last_verified_at"] = last_verified_at
    return fact


# ---------------------------------------------------------------------------
# read_jsonl
# ---------------------------------------------------------------------------

class TestReadJsonl:
    def test_reads_one_record_per_line(self, tmp_path):
        path = tmp_path / "facts.jsonl"
        path.write_text(
            json.dumps({"a": 1}) + "\n" + json.dumps({"a": 2}) + "\n",
            encoding="utf-8",
        )

        result = read_jsonl(path)

        assert result == [(1, {"a": 1}), (2, {"a": 2})]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "facts.jsonl"
        path.write_text(json.dumps({"a": 1}) + "\n\n" + json.dumps({"a": 2}) + "\n", encoding="utf-8")

        result = read_jsonl(path)

        assert [line_number for line_number, _ in result] == [1, 3]

    def test_invalid_json_raises_value_error_with_line_number(self, tmp_path):
        path = tmp_path / "facts.jsonl"
        path.write_text(json.dumps({"a": 1}) + "\nnot json\n", encoding="utf-8")

        with pytest.raises(ValueError, match="line 2"):
            read_jsonl(path)


# ---------------------------------------------------------------------------
# create_destination_fact_map
# ---------------------------------------------------------------------------

class TestCreateDestinationFactMap:
    def test_groups_facts_by_destination(self):
        facts = [
            make_fact(fact_id="f1", destination_id="sarajevo"),
            make_fact(fact_id="f2", destination_id="mostar"),
            make_fact(fact_id="f3", destination_id="sarajevo"),
        ]

        result = create_destination_fact_map(facts)

        assert set(result.keys()) == {"sarajevo", "mostar"}
        assert [f["fact_id"] for f in result["sarajevo"]] == ["f1", "f3"]
        assert [f["fact_id"] for f in result["mostar"]] == ["f2"]

    def test_empty_input_returns_empty_map(self):
        assert create_destination_fact_map([]) == {}


# ---------------------------------------------------------------------------
# validate_required_fields
# ---------------------------------------------------------------------------

class TestValidateRequiredFields:
    def test_complete_fact_passes(self):
        assert validate_required_fields(1, make_fact()) is None

    def test_missing_field_reports_line_and_field_name(self):
        fact = make_fact()
        del fact["source_id"]

        error = validate_required_fields(7, fact)

        assert error is not None
        assert "line 7" in error
        assert "source_id" in error

    def test_multiple_missing_fields_all_named(self):
        fact = make_fact()
        del fact["source_id"]
        del fact["category"]

        error = validate_required_fields(1, fact)

        assert "source_id" in error
        assert "category" in error


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_collapses_multiple_spaces(self):
        assert normalize_text("Sarajevo   je   glavni  grad") == "Sarajevo je glavni grad"

    def test_strips_leading_and_trailing_whitespace(self):
        assert normalize_text("  Mostar  ") == "Mostar"

    def test_collapses_tabs_and_newlines(self):
        assert normalize_text("Banja\tLuka\nje grad") == "Banja Luka je grad"

    def test_nfc_normalizes_combining_characters(self):
        # "c" + combining caron (decomposed) should become precomposed "č"
        decomposed = "c\u030c"
        normalized = normalize_text(decomposed)
        assert normalized == "\u010d"  # precomposed "č"


# ---------------------------------------------------------------------------
# is_text_invalid
# ---------------------------------------------------------------------------

class TestIsTextInvalid:
    def test_text_within_bounds_is_valid(self):
        text = "A" * MIN_TEXT_LENGTH
        assert is_text_invalid(text) is False

    def test_text_at_max_length_is_valid(self):
        text = "A" * MAX_TEXT_LENGTH
        assert is_text_invalid(text) is False

    def test_text_shorter_than_min_is_invalid(self):
        text = "A" * (MIN_TEXT_LENGTH - 1)
        assert is_text_invalid(text) is True

    def test_text_longer_than_max_is_invalid(self):
        text = "A" * (MAX_TEXT_LENGTH + 1)
        assert is_text_invalid(text) is True


# ---------------------------------------------------------------------------
# validate_destination_id / validate_source_id
# ---------------------------------------------------------------------------

class TestValidateDestinationId:
    def test_known_destination_passes(self):
        assert validate_destination_id("sarajevo", {"sarajevo", "mostar"}) is None

    def test_unknown_destination_fails(self):
        error = validate_destination_id("unknown_place", {"sarajevo"})
        assert error is not None
        assert "unknown_place" in error

    def test_empty_destination_set_skips_check(self):
        # an empty set means "no registry loaded" -> check is skipped
        assert validate_destination_id("anything", set()) is None


class TestValidateSourceId:
    def test_known_source_passes(self):
        assert validate_source_id("src1", {"src1", "src2"}) is None

    def test_unknown_source_fails(self):
        error = validate_source_id("unknown_src", {"src1"})
        assert error is not None
        assert "unknown_src" in error

    def test_empty_source_set_skips_check(self):
        assert validate_source_id("anything", set()) is None


# ---------------------------------------------------------------------------
# deduplicate_destination_facts
# ---------------------------------------------------------------------------

class TestDeduplicateDestinationFacts:
    def test_no_duplicates_keeps_all_facts(self):
        facts = [make_fact(fact_id="f1", text="A" * 40), make_fact(fact_id="f2", text="B" * 40)]

        deduplicated, removed_count = deduplicate_destination_facts(facts)

        assert len(deduplicated) == 2
        assert removed_count == 0

    def test_duplicate_text_keeps_first_occurrence_only(self):
        facts = [
            make_fact(fact_id="f1", text="Same text here"),
            make_fact(fact_id="f2", text="Different text"),
            make_fact(fact_id="f3", text="Same text here"),
        ]

        deduplicated, removed_count = deduplicate_destination_facts(facts)

        assert removed_count == 1
        assert [f["fact_id"] for f in deduplicated] == ["f1", "f2"]

    def test_all_duplicates_leaves_only_one(self):
        facts = [make_fact(fact_id=f"f{i}", text="Repeated text") for i in range(5)]

        deduplicated, removed_count = deduplicate_destination_facts(facts)

        assert len(deduplicated) == 1
        assert removed_count == 4
        assert deduplicated[0]["fact_id"] == "f0"

    def test_empty_list_returns_empty(self):
        deduplicated, removed_count = deduplicate_destination_facts([])
        assert deduplicated == []
        assert removed_count == 0


# ---------------------------------------------------------------------------
# validate_destination_facts
# ---------------------------------------------------------------------------

class TestValidateDestinationFacts:
    def test_enough_facts_passes(self):
        facts = [make_fact(fact_id=f"f{i}") for i in range(MIN_FACT_PER_DESTINATION)]
        assert validate_destination_facts("sarajevo", facts) is None

    def test_too_few_facts_fails(self):
        facts = [make_fact(fact_id=f"f{i}") for i in range(MIN_FACT_PER_DESTINATION - 1)]

        error = validate_destination_facts("sarajevo", facts)

        assert error is not None
        assert "sarajevo" in error


# ---------------------------------------------------------------------------
# validate_facts (end to end)
# ---------------------------------------------------------------------------

def write_jsonl(tmp_path, facts):
    """Write a list of fact dicts to a temp .jsonl file and return its path."""
    path = tmp_path / "facts.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for fact in facts:
            f.write(json.dumps(fact) + "\n")
    return path


def build_valid_facts_for(destination_id, count=MIN_FACT_PER_DESTINATION):
    """Build `count` distinct, valid facts for one destination."""
    return [
        make_fact(
            fact_id=f"{destination_id}_{i}",
            destination_id=destination_id,
            text=f"Unique fact number {i} about {destination_id} with enough length padding here.",
        )
        for i in range(count)
    ]


class TestValidateFactsEndToEnd:
    def test_fully_valid_registry_succeeds(self, tmp_path):
        facts = build_valid_facts_for("sarajevo")
        path = write_jsonl(tmp_path, facts)

        report = validate_facts(destinations={"sarajevo"}, sources={"src1"}, facts_path=path)

        assert report.success is True
        assert report.fact_count == MIN_FACT_PER_DESTINATION
        assert report.facts_per_destination == {"sarajevo": MIN_FACT_PER_DESTINATION}

    def test_missing_field_is_critical_error(self, tmp_path):
        facts = build_valid_facts_for("sarajevo")
        del facts[0]["source_id"]
        path = write_jsonl(tmp_path, facts)

        report = validate_facts(destinations={"sarajevo"}, sources={"src1"}, facts_path=path)

        assert report.success is False
        assert report.critical_error_count >= 1

    def test_invalid_text_length_is_dropped_and_flagged_critical(self, tmp_path):
        facts = build_valid_facts_for("sarajevo")
        facts[0]["text"] = "too short"
        path = write_jsonl(tmp_path, facts)

        report = validate_facts(destinations={"sarajevo"}, sources={"src1"}, facts_path=path)

        # one fact dropped -> destination now below minimum -> critical error
        assert report.success is False
        assert report.facts_per_destination["sarajevo"] == MIN_FACT_PER_DESTINATION - 1

    def test_duplicate_text_removed_and_reported_as_warning_not_error(self, tmp_path):
        facts = build_valid_facts_for("sarajevo", count=MIN_FACT_PER_DESTINATION + 1)
        facts[-1]["text"] = facts[0]["text"]  # force one duplicate
        path = write_jsonl(tmp_path, facts)

        report = validate_facts(destinations={"sarajevo"}, sources={"src1"}, facts_path=path)

        assert report.success is True  # still enough facts after removing the duplicate
        assert report.facts_per_destination["sarajevo"] == MIN_FACT_PER_DESTINATION
        assert any("duplicate" in w.lower() for w in report.warnings)
        assert not any("duplicate" in e.lower() for e in report.outcome.errors)

    def test_unknown_destination_id_is_critical_error(self, tmp_path):
        facts = build_valid_facts_for("atlantis")
        path = write_jsonl(tmp_path, facts)

        report = validate_facts(destinations={"sarajevo"}, sources={"src1"}, facts_path=path)

        assert report.success is False

    def test_empty_destination_and_source_sets_skip_those_checks(self, tmp_path):
        facts = build_valid_facts_for("anywhere")
        path = write_jsonl(tmp_path, facts)

        report = validate_facts(destinations=set(), sources=set(), facts_path=path)

        assert report.success is True


def test_duplicate_fact_id_is_critical_error(tmp_path):
    facts = build_valid_facts_for("sarajevo")
    facts[1]["fact_id"] = facts[0]["fact_id"]
    path = write_jsonl(tmp_path, facts)

    report = validate_facts(
        destinations={"sarajevo"},
        sources={"src1"},
        facts_path=path,
    )

    assert report.success is False
    assert any(
        "duplicate fact_id" in error.lower()
        for error in report.outcome.errors
    )


def test_invalid_category_is_critical_error(tmp_path):
    facts = build_valid_facts_for("sarajevo")
    facts[0]["category"] = "shopping"
    path = write_jsonl(tmp_path, facts)

    report = validate_facts(
        destinations={"sarajevo"},
        sources={"src1"},
        facts_path=path,
    )

    assert report.success is False
    assert any(
        "unsupported category" in error.lower()
        for error in report.outcome.errors
    )


def test_is_dynamic_must_be_boolean(tmp_path):
    facts = build_valid_facts_for("sarajevo")
    facts[0]["is_dynamic"] = "false"
    path = write_jsonl(tmp_path, facts)

    report = validate_facts(
        destinations={"sarajevo"},
        sources={"src1"},
        facts_path=path,
    )

    assert report.success is False
    assert any(
        "is_dynamic must be a boolean" in error.lower()
        for error in report.outcome.errors
    )


def test_last_verified_at_must_be_valid_iso_date(tmp_path):
    facts = build_valid_facts_for("sarajevo")
    facts[0]["last_verified_at"] = "25-07-2026"
    path = write_jsonl(tmp_path, facts)

    report = validate_facts(
        destinations={"sarajevo"},
        sources={"src1"},
        facts_path=path,
    )

    assert report.success is False
    assert any(
        "last_verified_at must be a valid yyyy-mm-dd date" in error.lower()
        for error in report.outcome.errors
    )

def test_blank_required_field_is_critical_error(tmp_path):
    facts = build_valid_facts_for("sarajevo")
    facts[0]["source_id"] = "   "
    path = write_jsonl(tmp_path, facts)

    report = validate_facts(
        destinations={"sarajevo"},
        sources={"src1"},
        facts_path=path,
    )

    assert report.success is False
    assert any(
        "empty required fields" in error.lower()
        for error in report.outcome.errors
    )


def test_dynamic_fact_requires_valid_until_or_verification_note(tmp_path):
    facts = build_valid_facts_for("sarajevo")
    facts[0]["is_dynamic"] = True
    facts[0]["text"] = (
        "Radno vrijeme lokaliteta može se mijenjati tokom turističke sezone."
    )
    path = write_jsonl(tmp_path, facts)

    report = validate_facts(
        destinations={"sarajevo"},
        sources={"src1"},
        facts_path=path,
    )

    assert report.success is False
    assert any(
        "dynamic fact requires valid_until or a verification note"
        in error.lower()
        for error in report.outcome.errors
    )


def test_fact_source_must_be_approved(tmp_path):
    facts = build_valid_facts_for("sarajevo")
    path = write_jsonl(tmp_path, facts)

    report = validate_facts(
        destinations={"sarajevo"},
        sources={"src1"},
        approved_sources=set(),
        facts_path=path,
    )

    assert report.success is False
    assert any(
        "source_id src1 is not approved" in error.lower()
        for error in report.outcome.errors
    )


def test_destination_missing_all_facts_is_critical_error(tmp_path):
    facts = build_valid_facts_for("sarajevo")
    path = write_jsonl(tmp_path, facts)

    report = validate_facts(
        destinations={"sarajevo", "mostar"},
        sources={"src1"},
        approved_sources={"src1"},
        facts_path=path,
    )

    assert report.success is False
    assert report.facts_per_destination["mostar"] == 0
    assert any(
        "mostar has 0 valid facts" in error.lower()
        for error in report.outcome.errors
    )


def test_report_cli_alias_is_supported(tmp_path):
    from src.bih_guide.data.validate_facts import build_parser

    report_path = tmp_path / "report.json"
    args = build_parser().parse_args(
        ["--report", str(report_path)]
    )

    assert args.report_out == report_path
