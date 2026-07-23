"""
Validator for the destination registry (destination_registry.csv).

Expected format for each line in the CSV:
    identifier,name,region,entity_type,priority

Where the allowed values are defined by the Region, EntityType and Priority enums.
"""

from enum import Enum
from typing import List, Set, Dict
import re


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
            NotImplementedError: If the label does not match any known region.
        """
        if label == "Sarajevo i okolina":
            return Region.SARAJEVO
        elif label == "Hercegovina i jugozapad":
            return Region.HERZEGOVINA
        elif label == "Srednja i sjeverna Bosna":
            return Region.MIDDLE_NORTH_BOSNIA
        elif label == "Sjeveroistočna i istočna Bosna":
            return Region.NORTHEAST_NORTH_BOSNIA
        elif label == "Krajina i zapadna Bosna":
            return Region.WEST_BOSNIA
        else:
            raise NotImplementedError(f"{label} is not defined for region.")


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
            NotImplementedError: If the label does not match any known entity type.
        """
        if label == "city":
            return EntityType.CITY
        elif label == "town":
            return EntityType.TOWN
        elif label == "nature_area":
            return EntityType.NATURE_AREA
        elif label == "mixed":
            return EntityType.MIXED
        else:
            raise NotImplementedError(f"{label} is not defined for entity type.")


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
            NotImplementedError: If the label does not match any known priority.
        """
        if label == "core":
            return Priority.CORE
        elif label == "secondary":
            return Priority.SECONDARY
        else:
            raise NotImplementedError(f"{label} is not defined for priority.")


class DestinationValidatorResult:
    """
    Outcome of a single validation step.

    Attributes:
        msg: Description of the result.
        success: Indicator whether the validation passed.
    """

    def __init__(self, msg: str, success: bool):
        """
        Initialize a validation result.

        Args:
            msg: Description of the result.
            success: Indicator whether the validation passed.
        """
        self.msg = msg
        self.success = success

    @staticmethod
    def ok() -> "DestinationValidatorResult":
        """
        Build a successful validation result.

        Returns:
            A DestinationValidatorResult with success=True.
        """
        return DestinationValidatorResult("Validation successful.", True)

    @staticmethod
    def error(msg: str) -> "DestinationValidatorResult":
        """
        Build a failed validation result.

        Args:
            msg: Description of why validation failed.

        Returns:
            A DestinationValidatorResult with success=False.
        """
        return DestinationValidatorResult(msg, False)


def validate_number_lines(count: int, target: int) -> DestinationValidatorResult:
    """
    Check that the total number of destination lines matches the target.

    Args:
        count: The actual number of lines found.
        target: The expected number of lines.

    Returns:
        DestinationValidatorResult.ok() if count == target, otherwise an
        error result describing the mismatch.
    """
    return (
        DestinationValidatorResult.ok()
        if count == target
        else DestinationValidatorResult.error(f"Number of destinations must be {target}")
    )


def validate_core_count(count: int, target: int) -> DestinationValidatorResult:
    """
    Check that the number of 'core' destinations matches the target.

    Args:
        count: The actual number of core destinations found.
        target: The expected number of core destinations.

    Returns:
        DestinationValidatorResult.ok() if count == target, otherwise an
        error result describing the mismatch.
    """
    return (
        DestinationValidatorResult.ok()
        if count == target
        else DestinationValidatorResult.error(f"Number of core destinations must be {target}")
    )


def validate_secondary_count(count: int, target: int) -> DestinationValidatorResult:
    """
    Check that the number of 'secondary' destinations matches the target.

    Args:
        count: The actual number of secondary destinations found.
        target: The expected number of secondary destinations.

    Returns:
        DestinationValidatorResult.ok() if count == target, otherwise an
        error result describing the mismatch.
    """
    return (
        DestinationValidatorResult.ok()
        if count == target
        else DestinationValidatorResult.error(f"Number of secondary destinations must be {target}")
    )


def validate_line_format(line_number: int, parts: List[str]) -> DestinationValidatorResult:
    """
    Check that a CSV line has exactly 5 comma-separated fields.

    Args:
        line_number: Index of the line being validated (used in error messages).
        parts: The line already split on commas.

    Returns:
        DestinationValidatorResult.ok() if the field count is correct,
        otherwise an error result noting whether fields are missing or extra.
    """
    parts_len = len(parts)

    if parts_len != LINE_PARTS:
        not_enough = parts_len < LINE_PARTS
        msg = (
            f"Some information missing on line {line_number}"
            if not_enough
            else f"Too much information on line {line_number}"
        )
        return DestinationValidatorResult.error(msg)

    return DestinationValidatorResult.ok()


def validate_identifier(identifier: str, destination_ids: Set[str]) -> DestinationValidatorResult:
    """
    Check that a destination identifier is well-formed and unique.

    A valid identifier contains only lowercase letters and underscores, and
    must start and end with a lowercase letter.

    Args:
        identifier: The identifier to validate.
        destination_ids: Set of identifiers already seen so far; used to detect duplicates.

    Returns:
        DestinationValidatorResult.ok() if the identifier is valid and not
        already present in destination_ids, otherwise an error result.
    """
    if not re.fullmatch(IDENTIFIER_PATTERN, identifier):
        return DestinationValidatorResult.error(
            "Destination identifier must start with a lowercase letter, "
            "end with a lowercase letter, and contain only lowercase letters and underscores."
        )

    if identifier in destination_ids:
        return DestinationValidatorResult.error(f"Duplicate destination identifier {identifier}.")

    return DestinationValidatorResult.ok()


def validate_region_type_priority(
    region: str, entity_type: str, priority: str
) -> DestinationValidatorResult:
    """
    Check that region, entity type, and priority are all recognized values.

    Args:
        region: Raw region string from the CSV.
        entity_type: Raw entity type string from the CSV.
        priority: Raw priority string from the CSV.

    Returns:
        DestinationValidatorResult.ok() if all three values map to a known
        enum member, otherwise an error result describing the first
        unrecognized value.
    """
    try:
        Region.from_str(region)
        EntityType.from_str(entity_type)
        Priority.from_str(priority)
        return DestinationValidatorResult.ok()
    except NotImplementedError as e:
        return DestinationValidatorResult.error(str(e))


def validate_destination_line(
    line_number: int,
    line: str,
    destination_ids: Set[str],
    counts: Dict[str, int],
) -> DestinationValidatorResult:
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
        DestinationValidatorResult.ok() if the line is fully valid, otherwise
        the first error result encountered.
    """
    parts = line.strip().split(",")
    result = validate_line_format(line_number, parts)
    if not result.success:
        return result

    identifier, name, region, entity_type, priority = parts[0], parts[1], parts[2], parts[3], parts[4]

    result = validate_identifier(identifier, destination_ids)
    if not result.success:
        return result
    destination_ids.add(identifier)

    result = validate_region_type_priority(region, entity_type, priority)
    if not result.success:
        return result

    if priority == Priority.CORE.value:
        counts["core"] += 1
    elif priority == Priority.SECONDARY.value:
        counts["secondary"] += 1

    return DestinationValidatorResult.ok()


def validate_destination_registry(destination_lines: List[str]) -> DestinationValidatorResult:
    """
    Validate the full destination registry.

    Checks, in order:
      1. The total line count matches DESTINATIONS_COUNT_TARGET.
      2. Every line individually passes validate_destination_line.
      3. The number of 'core' destinations matches CORE_DESTINATIONS_TARGET.
      4. The number of 'secondary' destinations matches SECONDARY_DESTINATIONS_TARGET.

    Args:
        destination_lines: All lines of the registry file, one destination
            per line.

    Returns:
        DestinationValidatorResult.ok() if the whole registry is valid,
        otherwise the first error result encountered.
    """
    result = validate_number_lines(len(destination_lines), DESTINATIONS_COUNT_TARGET)
    if not result.success:
        return result

    destination_ids: Set[str] = set()
    counts: Dict[str, int] = {"core": 0, "secondary": 0}

    for i, line in enumerate(destination_lines):
        result = validate_destination_line(i, line, destination_ids, counts)
        if not result.success:
            return result

    result = validate_core_count(counts["core"], CORE_DESTINATIONS_TARGET)
    if not result.success:
        return result

    result = validate_secondary_count(counts["secondary"], SECONDARY_DESTINATIONS_TARGET)
    if not result.success:
        return result

    return DestinationValidatorResult.ok()
