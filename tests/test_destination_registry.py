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
    return f"{identifier},{name},{region},{entity_type},{priority}"


# ----------------------------------------------------------------------
# Region
# ----------------------------------------------------------------------


class TestRegion:

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("Sarajevo i okolina", Region.SARAJEVO),
            ("Hercegovina i jugozapad", Region.HERZEGOVINA),
            ("Srednja i sjeverna Bosna", Region.MIDDLE_NORTH_BOSNIA),
            ("Sjeveroistočna i istočna Bosna", Region.NORTHEAST_NORTH_BOSNIA),
            ("Krajina i zapadna Bosna", Region.WEST_BOSNIA),
        ],
    )
    def test_valid_region(self, label, expected):
        assert Region.from_str(label) is expected

    def test_invalid_region(self):
        with pytest.raises(ValueError):
            Region.from_str("invalid")


# ----------------------------------------------------------------------
# EntityType
# ----------------------------------------------------------------------


class TestEntityType:

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("city", EntityType.CITY),
            ("town", EntityType.TOWN),
            ("nature_area", EntityType.NATURE_AREA),
            ("mixed", EntityType.MIXED),
        ],
    )
    def test_valid_entity_type(self, label, expected):
        assert EntityType.from_str(label) is expected

    def test_invalid_entity_type(self):
        with pytest.raises(ValueError):
            EntityType.from_str("invalid")


# ----------------------------------------------------------------------
# Priority
# ----------------------------------------------------------------------


class TestPriority:

    @pytest.mark.parametrize(
        "label,expected",
        [
            ("core", Priority.CORE),
            ("secondary", Priority.SECONDARY),
        ],
    )
    def test_valid_priority(self, label, expected):
        assert Priority.from_str(label) is expected

    def test_invalid_priority(self):
        with pytest.raises(ValueError):
            Priority.from_str("invalid")


# ----------------------------------------------------------------------
# Counts
# ----------------------------------------------------------------------


class TestCounts:

    def test_validate_number_lines(self):
        assert validate_number_lines(DESTINATIONS_COUNT_TARGET) is None
        assert validate_number_lines(10) is not None

    def test_validate_core_count(self):
        assert validate_core_count(CORE_DESTINATIONS_TARGET) is None
        assert validate_core_count(10) is not None

    def test_validate_secondary_count(self):
        assert validate_secondary_count(SECONDARY_DESTINATIONS_TARGET) is None
        assert validate_secondary_count(10) is not None


# ----------------------------------------------------------------------
# Line format
# ----------------------------------------------------------------------


class TestLineFormat:

    def test_valid_line(self):
        parts = ["id", "name", "region", "type", "priority"]
        assert validate_line_format(0, parts) is None

    def test_missing_fields(self):
        result = validate_line_format(2, ["id", "name"])

        assert result is not None
        assert "missing" in result.lower()

    def test_extra_fields(self):
        result = validate_line_format(
            2,
            ["id", "name", "region", "type", "priority", "extra"],
        )

        assert result is not None
        assert "too much" in result.lower()


# ----------------------------------------------------------------------
# Identifier
# ----------------------------------------------------------------------


class TestIdentifier:

    @pytest.mark.parametrize(
        "identifier",
        [
            "a",
            "sarajevo",
            "banja_luka",
            "np_sutjeska",
        ],
    )
    def test_valid_identifier(self, identifier):
        assert validate_identifier(identifier, set()) is None

    @pytest.mark.parametrize(
        "identifier",
        [
            "",
            "Sarajevo",
            "_sarajevo",
            "sarajevo_",
            "sarajevo1",
            "1sarajevo",
        ],
    )
    def test_invalid_identifier(self, identifier):
        assert validate_identifier(identifier, set()) is not None

    def test_duplicate_identifier(self):
        assert (
            validate_identifier("sarajevo", {"sarajevo"})
            is not None
        )

    def test_validator_does_not_modify_set(self):
        ids = set()

        validate_identifier("mostar", ids)

        assert ids == set()


# ----------------------------------------------------------------------
# Region / type / priority
# ----------------------------------------------------------------------


class TestRegionTypePriority:

    def test_valid_values(self):
        assert (
            validate_region_type_priority(
                "Sarajevo i okolina",
                "city",
                "core",
            )
            is None
        )

    @pytest.mark.parametrize(
        "region,entity,priority",
        [
            ("invalid", "city", "core"),
            ("Sarajevo i okolina", "invalid", "core"),
            ("Sarajevo i okolina", "city", "invalid"),
        ],
    )
    def test_invalid_values(
        self,
        region,
        entity,
        priority,
    ):
        assert (
            validate_region_type_priority(
                region,
                entity,
                priority,
            )
            is not None
        )


# ----------------------------------------------------------------------
# Destination line
# ----------------------------------------------------------------------


class TestDestinationLine:

    def test_valid_core_line(self):
        ids = set()
        counts = {"core": 0, "secondary": 0}

        assert (
            validate_destination_line(
                0,
                make_line(),
                ids,
                counts,
            )
            is None
        )

        assert ids == {"sarajevo"}
        assert counts == {
            "core": 1,
            "secondary": 0,
        }

    def test_valid_secondary_line(self):
        ids = set()
        counts = {"core": 0, "secondary": 0}

        assert (
            validate_destination_line(
                0,
                make_line(
                    identifier="mostar",
                    priority="secondary",
                ),
                ids,
                counts,
            )
            is None
        )

        assert counts == {
            "core": 0,
            "secondary": 1,
        }

    def test_duplicate_identifier(self):
        ids = set()
        counts = {"core": 0, "secondary": 0}

        validate_destination_line(
            0,
            make_line(),
            ids,
            counts,
        )

        assert (
            validate_destination_line(
                1,
                make_line(),
                ids,
                counts,
            )
            is not None
        )

    def test_invalid_line(self):
        ids = set()
        counts = {"core": 0, "secondary": 0}

        assert (
            validate_destination_line(
                0,
                "sarajevo,Sarajevo",
                ids,
                counts,
            )
            is not None
        )

        assert counts == {
            "core": 0,
            "secondary": 0,
        }

    def test_invalid_priority(self):
        ids = set()
        counts = {"core": 0, "secondary": 0}

        assert (
            validate_destination_line(
                0,
                make_line(priority="optional"),
                ids,
                counts,
            )
            is not None
        )

        assert counts == {
            "core": 0,
            "secondary": 0,
        }

# ----------------------------------------------------------------------
# Registry helpers
# ----------------------------------------------------------------------


def index_to_identifier(index: int) -> str:
    result = ""

    while True:
        index, remainder = divmod(index, 26)
        result = chr(ord("a") + remainder) + result

        if index == 0:
            break

        index -= 1

    return result


def build_registry(
    core_count=CORE_DESTINATIONS_TARGET,
    secondary_count=SECONDARY_DESTINATIONS_TARGET,
):
    lines = []

    for i in range(core_count):
        lines.append(
            make_line(
                identifier=f"core_{index_to_identifier(i)}",
                priority="core",
            )
        )

    for i in range(secondary_count):
        lines.append(
            make_line(
                identifier=f"secondary_{index_to_identifier(i)}",
                priority="secondary",
            )
        )

    return lines


# ----------------------------------------------------------------------
# Destination registry
# ----------------------------------------------------------------------


class TestDestinationRegistry:

    def test_valid_registry(self):
        registry = build_registry()

        report = validate_destination_registry(registry)

        assert report.success
        assert report.destination_count == DESTINATIONS_COUNT_TARGET
        assert report.core_count == CORE_DESTINATIONS_TARGET
        assert report.secondary_count == SECONDARY_DESTINATIONS_TARGET

    def test_invalid_total_count(self):
        registry = build_registry(
            core_count=CORE_DESTINATIONS_TARGET - 1,
        )

        report = validate_destination_registry(registry)

        assert not report.success
        assert str(DESTINATIONS_COUNT_TARGET) in report.msg

    @pytest.mark.parametrize(
        "core,secondary",
        [
            (
                CORE_DESTINATIONS_TARGET - 1,
                SECONDARY_DESTINATIONS_TARGET + 1,
            ),
            (
                CORE_DESTINATIONS_TARGET + 1,
                SECONDARY_DESTINATIONS_TARGET - 1,
            ),
        ],
    )
    def test_invalid_priority_counts(
        self,
        core,
        secondary,
    ):
        registry = build_registry(
            core_count=core,
            secondary_count=secondary,
        )

        report = validate_destination_registry(registry)

        assert not report.success

    def test_duplicate_identifier(self):
        registry = build_registry()

        registry[-1] = make_line(
            identifier="core_a",
            priority="secondary",
        )

        report = validate_destination_registry(registry)

        assert not report.success
        assert "Duplicate" in report.msg

    def test_registry_can_be_reused(self):
        registry = build_registry()

        first = validate_destination_registry(registry)
        second = validate_destination_registry(registry)

        assert first.success
        assert second.success

    def test_multiple_errors_are_collected(self):
        registry = build_registry(
            core_count=CORE_DESTINATIONS_TARGET - 1,
        )

        registry[0] = "invalid"

        report = validate_destination_registry(registry)

        assert not report.success
        assert len(report.outcome.errors) >= 2
        