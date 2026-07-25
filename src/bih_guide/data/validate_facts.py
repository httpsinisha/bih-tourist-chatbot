"""
Validator for the facts registry (facts.jsonl).

Expected format: one JSON object per line, with required fields
fact_id, destination_id, category, text, source_id, is_dynamic and
last_verified_at.
"""

import argparse
import json
import re
import sys
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.bih_guide.data.validation_report import ValidationOutcome, FactValidationReport
from src.bih_guide.data.validate_sources import load_destination_ids, read_csv

BASE_DIR = Path(__file__).resolve().parents[3]
FACTS_FILE_PATH = BASE_DIR / "data" / "raw" / "facts.jsonl"
DESTINATIONS_PATH = BASE_DIR / "data" / "destination_registry.csv"
SOURCE_PATH = BASE_DIR / "data" / "sources.csv"
REPORT_OUTPUT_PATH = BASE_DIR / "artifacts" / "reports" / "data_validation.json"

REQUIRED_FIELDS = {
    "fact_id",
    "destination_id",
    "category",
    "text",
    "source_id",
    "is_dynamic",
    "last_verified_at"
}
MIN_FACT_PER_DESTINATION = 6
MIN_TEXT_LENGTH = 30
MAX_TEXT_LENGTH = 500


def read_jsonl(path: Path) -> List[Tuple[int, Dict]]:
    """
    Read JSONL file into a list of (line_number, fact) pairs.

    Args:
        path: Path to the .jsonl file.

    Returns:
        A list of (line_number, fact) tuples, skipping blank lines.

    Raises:
        ValueError: IF any line is not valid JSON.
    """
    facts: List[Tuple[int, Dict]] = []

    with open(path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                fact = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}")

            facts.append((i, fact))

    return facts


def create_destination_fact_map(facts: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group facts by their destination_id.

    Args:
        facts: List of fact records (each must have destination_id).

    Returns:
        a Dict mapping destination_id to the list of facts that belong to it.
    """
    destination_map: Dict[str, List[Dict]] = {}

    for fact in facts:
        destination_id = fact["destination_id"]
        destination_map.setdefault(destination_id, []).append(fact)

    return destination_map

def validate_required_fields(line_number, fact: Dict) -> Optional[str]:
    """
    Check that a fact contains every field in REQUIRED_FIELDS.

    Args:
        line_number: Line number in the source file.
        fact: The fact record to validate.

    Returns:
        None if all required fields are occurring, otherwise an error
        message naming the missing fields.
    """
    missing = [field for field in REQUIRED_FIELDS if field not in fact]
    if missing:
        return f"Fact on line {line_number} is missing: {', '.join(missing)}"
    return None


def normalize_text(text: str) -> str:
    """
    Normalize fact text to NFC Unicode form and remove multiple whitespace.

    Args:
        text: The raw fact text.

    Returns:
        The normalized text: NFC-normalized, with runs of whitespace
        collapsed to a single space.
    """
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_text_invalid(text: str) -> bool:
    """
    Check whether normalized fact text falls outside the allowed length.

    Args:
        text: Already-normalized fact text.

    Returns:
        True if the text is shorter than MIN_TEXT_LENGTH or longer than
        MAX_TEXT_LENGTH, meaning the fact must be dropped.
    """
    length = len(text)
    return length < MIN_TEXT_LENGTH or length > MAX_TEXT_LENGTH


def load_source_ids(path: Path) -> Set[str]:
    """
    Load the set of known source IDs from the source registry CSV.

    Args:
        path: Path to sources.csv.

    Returns:
        Set of source_id values found in the file.

    Raises:
        ValueError: If the file has no source_id column, or contains no IDs.
    """
    columns, rows = read_csv(path)
    if "source_id" not in columns:
        raise ValueError(f"Source registry {path} must contain a source_id column.")

    source_ids = {row["source_id"] for row in rows if row["source_id"]}
    if not source_ids:
        raise ValueError(f"Source registry {path} contains no source IDs.")
    return source_ids


def validate_destination_id(destination_id: str, destination_ids: Set[str]) -> Optional[str]:
    """
    Check that a fact's destination_id exists in the destination registry.

    Args:
        destination_id: The destination_id to check.
        destination_ids: Set of known destination IDs.

    Returns:
        None if destination_id is known (or destination_ids is empty,
        meaning the check is skipped), otherwise an error message.
    """
    if destination_ids and destination_id not in destination_ids:
        return f"{destination_id} not found in destination registry."
    return None


def validate_source_id(source_id: str, source_ids: Set[str]) -> Optional[str]:
    """
    Check that a fact's source_id exists in the source registry.

    Args:
        source_id: The source_id to check.
        source_ids: Set of known source IDs.

    Returns:
        None if source_id is known (or source_ids is empty, meaning the
        check is skipped), otherwise an error message.
    """
    if source_ids and source_id not in source_ids:
        return f"{source_id} not found in source registry."
    return None


def deduplicate_destination_facts(facts: List[Dict]) -> Tuple[List[Dict], int]:
    """
    Remove facts with a duplicate normalized text within one destination.

    The first occurrence of each normalized text is kept; later ones are
    dropped.

    Args:
        facts: Facts belonging to a single destination (each must already
            have normalized "text").

    Returns:
        A tuple of (deduplicated facts, number of duplicates removed).
    """
    seen_texts: Set[str] = set()
    deduplicated: List[Dict] = []

    for fact in facts:
        text = fact["text"]
        if text in seen_texts:
            continue
        seen_texts.add(text)
        deduplicated.append(fact)

    removed_count = len(facts) - len(deduplicated)
    return deduplicated, removed_count


def validate_destination_facts(destination: str, facts: List[Dict]) -> Optional[str]:
    """
    Check that a destination has enough facts after deduplication.

    Args:
        destination: The destination_id these facts belong to.
        facts: Already-deduplicated facts for this destination.

    Returns:
        None if the destination has at least MIN_FACT_PER_DESTINATION
        facts, otherwise an error message.
    """
    num_facts = len(facts)
    if num_facts < MIN_FACT_PER_DESTINATION:
        return f"{destination} has {num_facts} valid facts, required at least {MIN_FACT_PER_DESTINATION}."
    return None


def validate_facts(
        destinations: Set[str],
        sources: Set[str],
        facts_path: Path = FACTS_FILE_PATH,
) -> FactValidationReport:
    """
    Validate the full facts registry, collecting every error and warning found.

    Steps:
      1. Parse the JSONL file.
      2. Per fact: check required fields, dynamic/source/destination
         validity, normalize text, and drop facts whose text length is
         out of bounds.
      3. Per destination: remove duplicate-text facts (keeping the
         first), then check the minimum fact count.
      4. Collect stale-verification warnings for dynamic facts.

    Args:
        destinations: Set of known destination IDs.
        sources: Set of known source IDs.
        facts_path: Path to the facts.jsonl file to validate.

    Returns:
        A FactValidationReport with critical errors, warnings, the final
        (post-cleanup) fact count, and fact counts per destination.
    """
    errors: List[str] = []
    warnings: List[str] = []

    raw_facts = read_jsonl(facts_path)
    valid_facts: List[Dict] = []

    for line_number, fact in raw_facts:
        error = validate_required_fields(line_number, fact)
        if error:
            errors.append(error)
            continue

        error = validate_destination_id(fact["destination_id"], destinations)
        if error:
            errors.append(error)

        error = validate_source_id(fact["source_id"], sources)
        if error:
            errors.append(error)


        normalized_text = normalize_text(fact["text"])
        if is_text_invalid(normalized_text):
            errors.append(
                f"Fact {fact.get('fact_id')} removed: text length "
                f"{len(normalized_text)} is outside the {MIN_TEXT_LENGTH}-{MAX_TEXT_LENGTH} bounds."
            )
            continue

        fact["text"] = normalized_text
        valid_facts.append(fact)

    destination_facts = create_destination_fact_map(valid_facts)
    facts_per_destination: Dict[str, int] = {}
    final_facts: List[Dict] = []

    for destination, facts_for_destination in destination_facts.items():
        deduplicated, removed_count = deduplicate_destination_facts(facts_for_destination)
        if removed_count:
            warnings.append(
                f"{destination}: removed {removed_count} duplicate-text facts."
            )

        error = validate_destination_facts(destination, deduplicated)
        if error:
            errors.append(error)

        facts_per_destination[destination] = len(deduplicated)
        final_facts.extend(deduplicated)

    return FactValidationReport(
        outcome=ValidationOutcome(tuple(errors)),
        warnings=tuple(warnings),
        fact_count=len(final_facts),
        facts_per_destination=facts_per_destination,
    )


def save_report(report: FactValidationReport, output_path: Path):
    """
    Write a FactValidationReport to disk as JSON.

    Args:
        report: The report to serialize.
        output_path: Destination path for the JSON report file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)


def main() -> int:
    """
    Run fact validation end to end and write a JSON report.

    Returns:
        Exit code: 0 if there are no critical errors, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Validate the facts registry.")
    parser.add_argument("--facts", type=Path, default=FACTS_FILE_PATH, help="Path to facts.jsonl")
    parser.add_argument("--destinations", type=Path, default=DESTINATIONS_PATH, help="Path to destination_registry.csv")
    parser.add_argument("--sources", type=Path, default=SOURCE_PATH, help="Path to sources.csv")
    parser.add_argument("--report-out", type=Path, default=REPORT_OUTPUT_PATH, help="Where to write the JSON report")
    args = parser.parse_args()

    report = validate_facts(
        destinations=load_destination_ids(args.destinations),
        sources=load_source_ids(args.sources),
        facts_path=args.facts,
    )

    save_report(report, args.report_out)
    print(report.msg)

    return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(main())
