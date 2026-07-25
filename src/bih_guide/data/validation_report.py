from dataclasses import dataclass


@dataclass
class ValidationOutcome:
    """
    Generic pass/fail outcome shared by every validation report.
    """

    errors: tuple[str, ...]

    @property
    def success(self) -> bool:
        """Return True when no errors were collected."""
        return not self.errors


@dataclass
class SourceValidationReport:
    """
    Result of validating a complete source registry.
    """

    outcome: ValidationOutcome
    total_sources: int
    approved_sources: int
    covered_destinations: int
    destination_count: int

    @property
    def success(self) -> bool:
        """Return True when the registry is fully valid."""
        return self.outcome.success

    @property
    def msg(self) -> str:
        """Return a concise result message."""

        if self.success:
            return (
                "Validation successful: "
                f"{self.approved_sources} approved sources cover "
                f"{self.covered_destinations}/{self.destination_count} destinations."
            )
        return "\n".join(self.outcome.errors)


@dataclass
class DestinationValidationReport:
    """
    Result of validating a complete destination registry.
    """

    outcome: ValidationOutcome
    destination_count: int
    core_count: int
    secondary_count: int

    @property
    def success(self) -> bool:
        """Return True when the registry is fully valid."""
        return self.outcome.success

    @property
    def msg(self) -> str:
        """Return a concise result message."""

        if self.success:
            return (
                "Validation successful: "
                f"{self.destination_count} destinations "
                f"({self.core_count} core / {self.secondary_count} secondary)"
            )
        return "\n".join(self.outcome.errors)


@dataclass
class FactValidationReport:
    """
    Result of validating a complete fact registry.
    """

    outcome: ValidationOutcome
    warnings: tuple[str, ...]
    fact_count: int
    facts_per_destination: dict[str, int]

    @property
    def success(self) -> bool:
        """Return True when the registry is fully valid."""
        return self.outcome.success

    @property
    def critical_error_count(self) -> int:
        """Number of critical errors collected."""
        return len(self.outcome.errors)

    @property
    def warning_count(self) -> int:
        """Number of warnings collected."""
        return len(self.warnings)

    @property
    def msg(self) -> str:
        """Return a concise result message."""

        lines = []
        if self.success:
            lines.append(
                f"Validation successful: {self.fact_count} facts "
                f"{len(self.facts_per_destination)} destinations."
            )
        else:
            lines.append(f"Validation failed with {self.critical_error_count} errors:")
            lines.extend(self.outcome.errors)

        if self.warnings:
            lines.append(f"\n{self.warning_count} warningS:")
            lines.extend(self.warnings)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Serialize this report into a JSON dict.

        Returns:
            A dict with fact counts per destination, and separate lists
            for critical errors and warnings.
        """
        return {
            "success": self.success,
            "fact_count": self.fact_count,
            "facts_per_destination": self.facts_per_destination,
            "critical_errors": list(self.outcome.errors),
            "warnings": list(self.warnings),
        }