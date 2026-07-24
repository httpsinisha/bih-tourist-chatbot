"""
Validator for the destination registry (destination_registry.csv).

Expected format for each line in the CSV:
    identifier,name,region,entity_type,priority

Where the allowed values are defined by the Region, EntityType and Priority enums.
"""

from enum import Enum
from typing import List, Set, Dict, Optional
import re
import argparse
from pathlib import Path
from typing import Sequence
from src.bih_guide.data.validation_report import ValidationOutcome, DestinationValidationReport


DESTINATIONS_FILE_PATH = Path("data/destination_registry.csv")
CORE_DESTINATIONS_TARGET = 30
SECONDARY_DESTINATIONS_TARGET = 42
DESTINATIONS_COUNT_TARGET = 72
LINE_PARTS = 5
IDENTIFIER_PATTERN = r"[a-z](?:[a-z_]*[a-z])?"


class Region(Enum):
    """
    Geographic region a destination belongs to.
    """

    SARAJEVO = "Sarajevo i okolina"
    HERZEGOVINA = "Hercegovina i jugozapad"
    MIDDLE_NORTH_BOSNIA = "Srednja i sjeverna Bosna"
    NORTHEAST_NORTH_BOSNIA = "Sjeveroistočna i istočna Bosna"
    WEST_BOSNIA = "Krajina i zapadna Bosna"

    @staticmethod
    def from_str(label: str) -> "Region":
        """
        Map a region label (as found in the CSV) to a Region member.

        Args:
            label: The raw region string from the CSV file.

        Returns:
            The matching Region enum member.

        Raises:
            ValueError: If the label does not match any known region.
        """
        return Region(label)



class EntityType(Enum):
    """
    Type of destination: city, town, nature area, or mixed.
    """

    CITY = "city"
    TOWN = "town"
    NATURE_AREA = "nature_area"
    MIXED = "mixed"

    @staticmethod
    def from_str(label: str) -> "EntityType":
        """
        Map an entity type label (as found in the CSV) to an EntityType member.

        Args:
            label: The raw entity type string from the CSV file.

        Returns:
            The matching EntityType enum member.

        Raises:
            ValueError: If the label does not match any known entity type.
        """
        return EntityType(label)


class Priority(Enum):
    """
    Destination priority: core (must-see) or secondary (nice-to-have).
    """

    CORE = "core"
    SECONDARY = "secondary"

    @staticmethod
    def from_str(label: str) -> "Priority":
        """
        Map a priority label (as found in the CSV) to a Priority member.

        Args:
            label: The raw priority string from the CSV file.

        Returns:
            The matching Priority enum member.

        Raises:
            ValueError: If the label does not match any known priority.
        """
        return Priority(label)


def validate_number_lines(count: int) -> Optional[str]:
    """
    Check that the total number of destination lines matches the target.

    Args:
        count: The actual number of lines found.

    Returns:
        None if count == target, otherwise str as error message.
    """
    if count == DESTINATIONS_COUNT_TARGET:
        return None
    return f"Number of destinations must be {DESTINATIONS_COUNT_TARGET}, found {count}."


def validate_core_count(count: int) -> Optional[str]:
    """
    Check that the number of 'core' destinations matches the target.

    Args:
        count: The actual number of core destinations found.

    Returns:
        None if count == target, otherwise str as error message.
    """
    if count == CORE_DESTINATIONS_TARGET:
        return None
    return f"Number of core destinations must be {CORE_DESTINATIONS_TARGET}, found {count}."


def validate_secondary_count(count: int) -> Optional[str]:
    """
    Check that the number of 'secondary' destinations matches the target.

    Args:
        count: The actual number of secondary destinations found.

    Returns:
        None if count == target, otherwise str as error message.
    """
    if count == SECONDARY_DESTINATIONS_TARGET:
        return None
    return f"Number of secondary destinations must be {SECONDARY_DESTINATIONS_TARGET}, found {count}."


def validate_line_format(line_number: int, parts: List[str]) -> Optional[str]:
    """
    Check that a CSV line has exactly 5 comma-separated fields.

    Args:
        line_number: Index of the line being validated (used in error messages).
        parts: The line already split on commas.

    Returns:
        None if the field count is correct,
        otherwise an error message noting whether fields are missing or extra.
    """
    parts_len = len(parts)

    if parts_len == LINE_PARTS:
        return None

    if parts_len < LINE_PARTS:
        return f"Some information missing on line {line_number}"
    return f"Too much information on line {line_number}"


def validate_identifier(identifier: str, destination_ids: Set[str]) -> Optional[str]:
    """
    Check that a destination identifier is well-formed and unique.

    A valid identifier contains only lowercase letters and underscores, and
    must start and end with a lowercase letter.

    Args:
        identifier: The identifier to validate.
        destination_ids: Set of identifiers already seen so far; used to detect duplicates.

    Returns:
        None if the identifier is valid and not already present in
        destination_ids, otherwise and error message.
    """
    if not re.fullmatch(IDENTIFIER_PATTERN, identifier):
        return (
            "Destination identifier must start with a lowercase letter, "
            "end with a lowercase letter, and contain only lowercase letters and underscores."
        )

    if identifier in destination_ids:
        return f"Duplicate destination identifier {identifier}"

    return None


def validate_region_type_priority(
    region: str, entity_type: str, priority: str
) -> Optional[str]:
    """
    Check that region, entity type, and priority are all recognized values.

    Args:
        region: Raw region string from the CSV.
        entity_type: Raw entity type string from the CSV.
        priority: Raw priority string from the CSV.

    Returns:
        None if all three values map to known enum member, otherwise an
        error message describing the first unrecognized value.
    """
    try:
        Region.from_str(region)
        EntityType.from_str(entity_type)
        Priority.from_str(priority)
        return None
    except ValueError as e:
        return str(e)


def validate_destination_line(
    line_number: int,
    line: str,
    destination_ids: Set[str],
    counts: Dict[str, int],
) -> Optional[str]:
    """
    Validate a single CSV line and update shared tracking state.

    Validates the line's field count, identifier (adding it to
    destination_ids on success), and region/entity_type/priority values.
    On success, increments counts['core'] or counts['secondary'] depending
    on the line's priority.

    Args:
        line_number: Index of the line being validated (used in error messages).
        line: The raw CSV line, e.g. "sarajevo,Sarajevo,Sarajevo i okolina,city,core".
        destination_ids: Set of identifiers seen so far across the registry;
            mutated in place when a new valid identifier is found.
        counts: Running tally with keys 'core' and 'secondary'; mutated in
            place when the line's priority is valid.

    Returns:
        None if the line is fully valid, otherwise
        the first error result encountered.
    """
    parts = line.strip().split(",")

    error = validate_line_format(line_number, parts)
    if error:
        return error

    identifier, name, region, entity_type, priority = parts[0], parts[1], parts[2], parts[3], parts[4]

    error = validate_identifier(identifier, destination_ids)
    if error:
        return error
    destination_ids.add(identifier)

    error = validate_region_type_priority(region, entity_type, priority)
    if error:
        return error

    if priority == Priority.CORE.value:
        counts["core"] += 1
    elif priority == Priority.SECONDARY.value:
        counts["secondary"] += 1

    return None

def validate_destination_registry(destination_lines: List[str]) -> DestinationValidationReport:
    """
    Validate the full destination registry, collecting every found error.

    Args:
        destination_lines: All lines of the registry file, one destination
            per line.

    Returns:
        A DestinationValidationReport describing the outcome, including
        every error message collected and the final destination/core/
        secondary counts.

    """
    errors: List[str] = []
    destination_count = len(destination_lines)

    count_error = validate_number_lines(destination_count)
    if count_error:
        errors.append(count_error)

    destination_ids: Set[str] = set()
    counts: Dict[str, int] = {"core": 0, "secondary": 0}

    for i, line in enumerate(destination_lines):
        line_error = validate_destination_line(i, line, destination_ids, counts)
        if line_error:
            errors.append(line_error)


    core_error = validate_core_count(counts["core"])
    if core_error:
        errors.append(core_error)

    secondary_error = validate_secondary_count(counts["secondary"])
    if secondary_error:
        errors.append(secondary_error)

    return DestinationValidationReport(
        outcome=ValidationOutcome(tuple(errors)),
        destination_count=destination_count,
        core_count=counts["core"],
        secondary_count=counts["secondary"]
    )

def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.
    """

    parser = argparse.ArgumentParser(
        description="Validate the destination registry."
    )
    parser.add_argument(
        "--destinations",
        type=Path,
        default=DESTINATIONS_FILE_PATH,
        help="Path to destination_registry.csv.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run validation and return a process exit code.
    """

    args = build_parser().parse_args(argv)

    try:
        with args.destinations.open(encoding="utf-8") as file:
            destination_lines = [
                line.strip()
                for line in file
                if line.strip()
            ]
    except OSError as exc:
        print(f"Validation failed: {exc}")
        return 1

    report = validate_destination_registry(destination_lines[1:])

    print(report.msg)
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
