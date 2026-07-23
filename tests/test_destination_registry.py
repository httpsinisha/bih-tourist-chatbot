"""
Tests for validate_destinations.py
"""

import pytest

from bih_guide.data.validate_destinations import (
    CORE_DESTINATIONS_TARGET,
    SECONDARY_DESTINATIONS_TARGET,
    DESTINATIONS_COUNT_TARGET,
    Region,
    EntityType,
    Priority,
    DestinationValidatorResult,
    validate_number_lines,
    validate_core_count,
    validate_secondary_count,
    validate_line_format,
    validate_identifier,
    validate_region_type_priority,
    validate_destination_line,
    validate_destination_registry,
)


def make_line(
    identifier="sarajevo",
    name="Sarajevo",
    region="Sarajevo i okolina",
    entity_type="city",
    priority="core",
):
    """Build a valid CSV line."""
    return f"{identifier},{name},{region},{entity_type},{priority}"


# ---------------------------------------------------------------------------
# Region / EntityType / Priority enums
# ---------------------------------------------------------------------------

class TestRegionFromStr:
    def test_valid_labels_map_to_expected_members(self):
        assert Region.from_str("Sarajevo i okolina") is Region.SARAJEVO
        assert Region.from_str("Hercegovina i jugozapad") is Region.HERZEGOVINA
        assert Region.from_str("Srednja i sjeverna Bosna") is Region.MIDDLE_NORTH_BOSNIA
        assert Region.from_str("Sjeveroistočna i istočna Bosna") is Region.NORTHEAST_NORTH_BOSNIA
        assert Region.from_str("Krajina i zapadna Bosna") is Region.WEST_BOSNIA

    def test_unknown_label_raises(self):
        with pytest.raises(NotImplementedError):
            Region.from_str("Nepostojeci region")


class TestEntityTypeFromStr:
    def test_valid_labels_map_to_expected_members(self):
        assert EntityType.from_str("city") is EntityType.CITY
        assert EntityType.from_str("town") is EntityType.TOWN
        assert EntityType.from_str("nature_area") is EntityType.NATURE_AREA
        assert EntityType.from_str("mixed") is EntityType.MIXED

    def test_unknown_label_raises(self):
        with pytest.raises(NotImplementedError):
            EntityType.from_str("village")


class TestPriorityFromStr:
    def test_valid_labels_map_to_expected_members(self):
        assert Priority.from_str("core") is Priority.CORE
        assert Priority.from_str("secondary") is Priority.SECONDARY

    def test_unknown_label_raises(self):
        with pytest.raises(NotImplementedError):
            Priority.from_str("optional")


# ---------------------------------------------------------------------------
# DestinationValidatorResult
# ---------------------------------------------------------------------------

class TestDestinationValidatorResult:
    def test_ok_is_successful(self):
        result = DestinationValidatorResult.ok()
        assert result.success is True
        assert result.msg == "Validation successful."

    def test_error_is_unsuccessful_and_carries_message(self):
        result = DestinationValidatorResult.error("something went wrong")
        assert result.success is False
        assert result.msg == "something went wrong"


# ---------------------------------------------------------------------------
# validate_number_lines
# ---------------------------------------------------------------------------

class TestValidateNumberLines:
    def test_matching_count_passes(self):
        assert validate_number_lines(72, 72).success is True

    def test_mismatched_count_fails_with_message(self):
        result = validate_number_lines(70, 72)
        assert result.success is False
        assert "72" in result.msg


# ---------------------------------------------------------------------------
# validate_core_count / validate_secondary_count
# ---------------------------------------------------------------------------

class TestValidateCoreCount:
    def test_matching_count_passes(self):
        assert validate_core_count(30, 30).success is True

    def test_mismatched_count_fails(self):
        assert validate_core_count(29, 30).success is False


class TestValidateSecondaryCount:
    def test_matching_count_passes(self):
        assert validate_secondary_count(42, 42).success is True

    def test_mismatched_count_fails(self):
        assert validate_secondary_count(41, 42).success is False


# ---------------------------------------------------------------------------
# validate_line_format
# ---------------------------------------------------------------------------

class TestValidateLineFormat:
    def test_exact_field_count_passes(self):
        parts = ["id", "name", "region", "type", "priority"]
        assert validate_line_format(0, parts).success is True

    def test_missing_fields_fails(self):
        parts = ["id", "name", "region"]
        result = validate_line_format(3, parts)
        assert result.success is False
        assert "missing" in result.msg.lower()
        assert "3" in result.msg

    def test_extra_fields_fails(self):
        parts = ["id", "name", "region", "type", "priority", "extra"]
        result = validate_line_format(5, parts)
        assert result.success is False
        assert "too much" in result.msg.lower()


# ---------------------------------------------------------------------------
# validate_identifier
# ---------------------------------------------------------------------------

class TestValidateIdentifier:
    def test_valid_single_letter_identifier_passes(self):
        assert validate_identifier("a", set()).success is True

    def test_valid_multi_letter_identifier_passes(self):
        assert validate_identifier("banja_luka", set()).success is True

    def test_uppercase_letters_fail(self):
        assert validate_identifier("Sarajevo", set()).success is False

    def test_leading_underscore_fails(self):
        assert validate_identifier("_sarajevo", set()).success is False

    def test_trailing_underscore_fails(self):
        assert validate_identifier("sarajevo_", set()).success is False

    def test_digits_fail(self):
        assert validate_identifier("sarajevo1", set()).success is False

    def test_empty_string_fails(self):
        assert validate_identifier("", set()).success is False

    def test_duplicate_identifier_fails(self):
        seen = {"sarajevo"}
        result = validate_identifier("sarajevo", seen)
        assert result.success is False
        assert "duplicate" in result.msg.lower()

    def test_new_identifier_not_mutated_into_set_by_validator(self):
        seen = set()
        validate_identifier("mostar", seen)
        assert "mostar" not in seen


# ---------------------------------------------------------------------------
# validate_region_type_priority
# ---------------------------------------------------------------------------

class TestValidateRegionTypePriority:
    def test_all_valid_passes(self):
        result = validate_region_type_priority("Sarajevo i okolina", "city", "core")
        assert result.success is True

    def test_invalid_region_fails(self):
        result = validate_region_type_priority("Nepostojeci region", "city", "core")
        assert result.success is False

    def test_invalid_entity_type_fails(self):
        result = validate_region_type_priority("Sarajevo i okolina", "village", "core")
        assert result.success is False

    def test_invalid_priority_fails(self):
        result = validate_region_type_priority("Sarajevo i okolina", "city", "optional")
        assert result.success is False


# ---------------------------------------------------------------------------
# validate_destination_line
# ---------------------------------------------------------------------------

class TestValidateDestinationLine:
    def test_valid_line_passes_and_updates_state(self):
        destination_ids = set()
        counts = {"core": 0, "secondary": 0}
        result = validate_destination_line(0, make_line(), destination_ids, counts)

        assert result.success is True
        assert "sarajevo" in destination_ids
        assert counts["core"] == 1
        assert counts["secondary"] == 0

    def test_secondary_priority_increments_secondary_count(self):
        destination_ids = set()
        counts = {"core": 0, "secondary": 0}

        line = make_line(identifier="mostar", priority="secondary")
        result = validate_destination_line(0, line, destination_ids, counts)

        assert result.success is True
        assert counts["core"] == 0
        assert counts["secondary"] == 1

    def test_malformed_line_fails_and_does_not_touch_counts(self):
        destination_ids = set()
        counts = {"core": 0, "secondary": 0}

        result = validate_destination_line(0, "sarajevo,Sarajevo", destination_ids, counts)

        assert result.success is False
        assert counts == {"core": 0, "secondary": 0}
        assert destination_ids == set()

    def test_duplicate_identifier_across_two_lines_fails_on_second(self):
        destination_ids = set()
        counts = {"core": 0, "secondary": 0}

        first = validate_destination_line(0, make_line(), destination_ids, counts)
        second = validate_destination_line(1, make_line(), destination_ids, counts)

        assert first.success is True
        assert second.success is False
        assert counts["core"] == 1

    def test_invalid_priority_does_not_increment_counts(self):
        destination_ids = set()
        counts = {"core": 0, "secondary": 0}

        line = make_line(priority="optional")
        result = validate_destination_line(0, line, destination_ids, counts)

        assert result.success is False
        assert counts == {"core": 0, "secondary": 0}

    def test_whitespace_around_line_is_stripped(self):
        destination_ids = set()
        counts = {"core": 0, "secondary": 0}
        result = validate_destination_line(0, "  " + make_line() + "  \n", destination_ids, counts)

        assert result.success is True


# ---------------------------------------------------------------------------
# validate_destination_registry
# ---------------------------------------------------------------------------

def index_to_identifier(index: int) -> str:
    result = ""

    while True:
        index, remainder = divmod(index, 26)
        result = chr(ord('a') + remainder) + result

        if index == 0:
            break

        index -= 1

    return result

def build_registry(core_count=CORE_DESTINATIONS_TARGET, secondary_count=SECONDARY_DESTINATIONS_TARGET):
    """Build a fully valid registry with the given core/secondary split."""
    lines = []
    for i in range(core_count):
        lines.append(make_line(identifier=f"core_dest_{index_to_identifier(i)}", priority="core"))
    for i in range(secondary_count):
        lines.append(make_line(identifier=f"secondary_dest_{index_to_identifier(i)}", priority="secondary"))
    return lines


class TestValidateDestinationRegistry:
    def test_fully_valid_registry_passes(self):
        lines = build_registry()
        print(lines[0])
        result = validate_destination_registry(lines)

        assert len(lines) == DESTINATIONS_COUNT_TARGET
        assert result.success is True

    def test_wrong_total_line_count_fails_before_per_line_checks(self):
        lines = build_registry(core_count=CORE_DESTINATIONS_TARGET - 1)
        result = validate_destination_registry(lines)
        assert result.success is False
        assert str(DESTINATIONS_COUNT_TARGET) in result.msg

    def test_too_few_core_destinations_fails(self):
        lines = build_registry(
            core_count=CORE_DESTINATIONS_TARGET - 1,
            secondary_count=SECONDARY_DESTINATIONS_TARGET + 1,
        )
        result = validate_destination_registry(lines)
        assert result.success is False

    def test_too_few_secondary_destinations_fails(self):
        lines = build_registry(
            core_count=CORE_DESTINATIONS_TARGET + 1,
            secondary_count=SECONDARY_DESTINATIONS_TARGET - 1,
        )
        result = validate_destination_registry(lines)
        assert result.success is False

    def test_duplicate_identifier_in_full_registry_fails(self):
        lines = build_registry()
        lines[-1] = make_line(identifier="core_dest_0", priority="secondary")
        result = validate_destination_registry(lines)

        assert result.success is False

    def test_registry_is_reusable_across_multiple_calls(self):
        lines = build_registry()

        first = validate_destination_registry(lines)
        second = validate_destination_registry(lines)

        assert first.success is True
        assert second.success is True
