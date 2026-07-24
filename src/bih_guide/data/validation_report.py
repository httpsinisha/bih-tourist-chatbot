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
