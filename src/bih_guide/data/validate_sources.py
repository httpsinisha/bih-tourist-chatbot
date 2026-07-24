"""Validate the tourism source registry used by the BiH tourist chatbot.

The default command can be run from the repository root::

    python -m bih_guide.data.validate_sources

It validates ``data/sources.csv`` against
``data/destination_registry.csv`` without making network requests. URL
availability is intentionally checked manually in a browser during source
review, while this module checks the registry's structure and consistency.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlparse
from src.bih_guide.data.validation_report import ValidationOutcome, SourceValidationReport

MIN_APPROVED_SOURCES = 90

EXPECTED_COLUMNS = (
    "source_id",
    "destination_id",
    "title",
    "url",
    "source_type",
    "retrieved_at",
    "status",
    "notes",
)

REQUIRED_FIELDS = (
    "source_id",
    "destination_id",
    "title",
    "url",
    "source_type",
    "retrieved_at",
    "status",
)

ALLOWED_SOURCE_TYPES = frozenset(
    {
        "official_tourism",
        "government",
        "institution",
        "unesco",
        "secondary_verified",
    }
)

ALLOWED_STATUSES = frozenset({"pending", "approved", "rejected", "unavailable"})


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a UTF-8 CSV file and return its columns and normalized rows."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        rows = [
            {key: (value or "").strip() for key, value in row.items() if key is not None}
            for row in reader
        ]
    return columns, rows


def load_destination_ids(path: Path) -> set[str]:
    """Load all destination IDs from ``destination_registry.csv``."""

    columns, rows = read_csv(path)
    if "destination_id" not in columns:
        raise ValueError(
            f"Destination registry {path} must contain a destination_id column."
        )

    destination_ids = {row["destination_id"] for row in rows if row["destination_id"]}
    if not destination_ids:
        raise ValueError(f"Destination registry {path} contains no destination IDs.")
    return destination_ids


def validate_columns(columns: Sequence[str]) -> list[str]:
    """Validate the exact source-registry column order."""

    if tuple(columns) == EXPECTED_COLUMNS:
        return []
    return [
        "sources.csv columns must be exactly: " + ",".join(EXPECTED_COLUMNS)
    ]


def is_valid_http_url(value: str) -> bool:
    """Return whether *value* is an absolute HTTP(S) URL."""

    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_valid_iso_date(value: str) -> bool:
    """Return whether *value* is a real date in YYYY-MM-DD form."""

    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return False
    return parsed.isoformat() == value


def validate_sources(
    rows: Iterable[Mapping[str, str]],
    destination_ids: set[str],
    *,
    min_approved: int = MIN_APPROVED_SOURCES,
) -> SourceValidationReport:
    """Validate source rows against the known destination registry.

    The validator checks required fields, unique source IDs and URLs, valid
    destinations, approved source coverage, allowed categorical values, URL
    syntax and ISO-formatted retrieval dates.
    """

    source_rows = list(rows)
    errors: list[str] = []
    seen_source_ids: set[str] = set()
    seen_urls: set[str] = set()
    approved_destinations: set[str] = set()
    approved_count = 0

    for line_number, row in enumerate(source_rows, start=2):
        missing_fields = [field for field in REQUIRED_FIELDS if not row.get(field, "").strip()]
        if missing_fields:
            errors.append(
                f"Line {line_number}: empty required fields: {', '.join(missing_fields)}."
            )

        source_id = row.get("source_id", "").strip()
        destination_id = row.get("destination_id", "").strip()
        url = row.get("url", "").strip()
        source_type = row.get("source_type", "").strip()
        retrieved_at = row.get("retrieved_at", "").strip()
        status = row.get("status", "").strip()

        if source_id:
            if source_id in seen_source_ids:
                errors.append(f"Line {line_number}: duplicate source_id {source_id}.")
            seen_source_ids.add(source_id)

        if destination_id and destination_id not in destination_ids:
            errors.append(
                f"Line {line_number}: unknown destination_id {destination_id}."
            )

        if url:
            normalized_url = url.rstrip("/")
            if not is_valid_http_url(url):
                errors.append(f"Line {line_number}: invalid URL {url}.")
            if normalized_url in seen_urls:
                errors.append(f"Line {line_number}: duplicate URL {url}.")
            seen_urls.add(normalized_url)

        if source_type and source_type not in ALLOWED_SOURCE_TYPES:
            errors.append(
                f"Line {line_number}: unsupported source_type {source_type}."
            )

        if status and status not in ALLOWED_STATUSES:
            errors.append(f"Line {line_number}: unsupported status {status}.")

        if retrieved_at and not is_valid_iso_date(retrieved_at):
            errors.append(
                f"Line {line_number}: retrieved_at must be a real YYYY-MM-DD date."
            )

        if status == "approved":
            approved_count += 1
            if destination_id in destination_ids:
                approved_destinations.add(destination_id)

    if approved_count < min_approved:
        errors.append(
            f"At least {min_approved} approved sources are required; found {approved_count}."
        )

    missing_destinations = sorted(destination_ids - approved_destinations)
    if missing_destinations:
        errors.append(
            "Destinations without an approved source: " + ", ".join(missing_destinations) + "."
        )

    return SourceValidationReport(
        outcome=ValidationOutcome(tuple(errors)),
        total_sources=len(source_rows),
        approved_sources=approved_count,
        covered_destinations=len(approved_destinations),
        destination_count=len(destination_ids),
    )


def validate_files(
    destinations_path: Path,
    sources_path: Path,
    *,
    min_approved: int = MIN_APPROVED_SOURCES,
) -> SourceValidationReport:
    """Load and validate both registry files."""

    destination_ids = load_destination_ids(destinations_path)
    columns, rows = read_csv(sources_path)
    column_errors = validate_columns(columns)
    report = validate_sources(rows, destination_ids, min_approved=min_approved)

    if not column_errors:
        return report

    return SourceValidationReport(
        outcome=ValidationOutcome(tuple(column_errors) + report.outcome.errors),
        total_sources=report.total_sources,
        approved_sources=report.approved_sources,
        covered_destinations=report.covered_destinations,
        destination_count=report.destination_count,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description="Validate the tourism source registry.")
    parser.add_argument(
        "--destinations",
        type=Path,
        default=Path("data/destination_registry.csv"),
        help="Path to destination_registry.csv.",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("data/sources.csv"),
        help="Path to sources.csv.",
    )
    parser.add_argument(
        "--min-approved",
        type=int,
        default=MIN_APPROVED_SOURCES,
        help="Minimum number of approved sources.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation and return a process exit code."""

    args = build_parser().parse_args(argv)

    try:
        report = validate_files(
            args.destinations,
            args.sources,
            min_approved=args.min_approved,
        )
    except (OSError, ValueError) as exc:
        print(f"Validation failed: {exc}")
        return 1

    print(report.msg)
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
