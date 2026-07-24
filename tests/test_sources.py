"""Tests for the tourism source registry validator."""

from __future__ import annotations

import csv
from pathlib import Path

from bih_guide.data.validate_sources import (
    EXPECTED_COLUMNS,
    SourceValidationReport,
    is_valid_http_url,
    is_valid_iso_date,
    load_destination_ids,
    validate_columns,
    validate_files,
    validate_sources,
)


def make_source(
    index: int,
    destination_id: str,
    *,
    status: str = "approved",
    url: str | None = None,
    source_type: str = "official_tourism",
    retrieved_at: str = "2026-07-24",
) -> dict[str, str]:
    """Build one valid source row."""

    return {
        "source_id": f"SRC-TEST-{index:03d}",
        "destination_id": destination_id,
        "title": f"Test source {index}",
        "url": url or f"https://example.org/source-{index}",
        "source_type": source_type,
        "retrieved_at": retrieved_at,
        "status": status,
        "notes": "",
    }


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    """Write a UTF-8 CSV test fixture."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_report_success_property_and_message():
    report = SourceValidationReport((), 90, 90, 72, 72)
    assert report.success is True
    assert "90 approved" in report.msg


def test_validate_columns_accepts_exact_schema():
    assert validate_columns(EXPECTED_COLUMNS) == []


def test_validate_columns_rejects_wrong_order_or_missing_column():
    assert validate_columns(EXPECTED_COLUMNS[:-1])
    assert validate_columns(tuple(reversed(EXPECTED_COLUMNS)))


def test_http_url_validation():
    assert is_valid_http_url("https://example.org/place") is True
    assert is_valid_http_url("http://example.org") is True
    assert is_valid_http_url("example.org/place") is False
    assert is_valid_http_url("ftp://example.org/place") is False


def test_iso_date_validation():
    assert is_valid_iso_date("2026-07-24") is True
    assert is_valid_iso_date("2026-02-30") is False
    assert is_valid_iso_date("24-07-2026") is False


def test_valid_registry_passes():
    destination_ids = {f"dest_{index}" for index in range(72)}
    rows = [make_source(index, destination_id) for index, destination_id in enumerate(destination_ids)]
    rows.extend(
        make_source(72 + index, f"dest_{index}") for index in range(18)
    )

    report = validate_sources(rows, destination_ids)

    assert report.success is True
    assert report.approved_sources == 90
    assert report.covered_destinations == 72


def test_registry_rejects_too_few_approved_sources():
    destination_ids = {"sarajevo"}
    rows = [make_source(1, "sarajevo")]

    report = validate_sources(rows, destination_ids)

    assert report.success is False
    assert "At least 90 approved" in report.msg


def test_registry_reports_destination_without_approved_source():
    destination_ids = {"sarajevo", "mostar"}
    rows = [make_source(index, "sarajevo") for index in range(90)]

    report = validate_sources(rows, destination_ids)

    assert report.success is False
    assert "mostar" in report.msg


def test_registry_rejects_unknown_destination():
    report = validate_sources(
        [make_source(1, "unknown")],
        {"sarajevo"},
        min_approved=1,
    )

    assert report.success is False
    assert "unknown destination_id" in report.msg


def test_registry_rejects_duplicate_source_id_and_url():
    first = make_source(1, "sarajevo")
    second = make_source(1, "sarajevo", url=first["url"] + "/")

    report = validate_sources(
        [first, second],
        {"sarajevo"},
        min_approved=1,
    )

    assert report.success is False
    assert "duplicate source_id" in report.msg
    assert "duplicate URL" in report.msg


def test_registry_rejects_invalid_fields():
    row = make_source(
        1,
        "sarajevo",
        source_type="blog",
        retrieved_at="2026-99-99",
    )
    row["status"] = "unknown"
    row["title"] = ""
    row["url"] = "not-a-url"

    report = validate_sources([row], {"sarajevo"}, min_approved=0)

    assert report.success is False
    assert "empty required fields" in report.msg
    assert "invalid URL" in report.msg
    assert "unsupported source_type" in report.msg
    assert "unsupported status" in report.msg
    assert "retrieved_at" in report.msg


def test_load_destination_ids_and_validate_files(tmp_path: Path):
    destinations_path = tmp_path / "destination_registry.csv"
    sources_path = tmp_path / "sources.csv"

    write_csv(
        destinations_path,
        ("destination_id", "name", "region", "entity_type", "priority"),
        [
            {
                "destination_id": "sarajevo",
                "name": "Sarajevo",
                "region": "Sarajevo i okolina",
                "entity_type": "city",
                "priority": "core",
            }
        ],
    )
    write_csv(sources_path, EXPECTED_COLUMNS, [make_source(1, "sarajevo")])

    assert load_destination_ids(destinations_path) == {"sarajevo"}
    report = validate_files(destinations_path, sources_path, min_approved=1)
    assert report.success is True
