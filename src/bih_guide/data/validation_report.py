from dataclasses import dataclass

@dataclass
class ValidationOutcome:
    errors: tuple[str, ...]

    @property
    def success(self) -> bool:
        return not self.errors


@dataclass
class SourceValidationReport:
    outcome: ValidationOutcome
    total_sources: int
    approved_sources: int
    covered_destinations: int
    destination_count: int

    @property
    def success(self) -> bool:
        return self.outcome.success

    @property
    def msg(self) -> str:
        if self.success:
            return (
                "Validation successful: "
                f"{self.approved_sources} approved sources cover "
                f"{self.covered_destinations}/{self.destination_count} destinations."
            )
        return "\n".join(self.outcome.errors)

