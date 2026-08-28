"""PHI detection for free-text fields reaching the API.

`config/settings.json` declared `presidio.enabled: true` while no
Presidio code existed anywhere in the repository. A configuration
asserting a safety control that is not implemented is worse than one
admitting the control is absent: it answers the question nobody then
asks again.

Two backends:

`patterns` (always available) matches direct identifiers -- national
insurance and social security numbers, emails, phone numbers, dates of
birth, and long digit runs of the shape medical record numbers take. It
is deliberately narrow, and its limits are stated rather than implied: it
does not find names, addresses, or an identifier embedded in prose. It
catches the realistic mistake, which is an MRN pasted into `patient_id`.

`presidio` uses Microsoft Presidio's NLP recognisers and is the optional
`phi` extra. Selecting it when it is not installed raises at startup
rather than falling back, because a silent fallback is exactly how a
config comes to claim a control it does not have.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


class PHIConfigurationError(RuntimeError):
    """PHI detection was requested in a form that cannot be provided."""


class PHIDetected(ValueError):
    """Free text carried something that looks like a direct identifier."""

    def __init__(self, findings: List["PHIFinding"]):
        self.findings = findings
        summary = ", ".join(f"{f.kind} in {f.field}" for f in findings)
        super().__init__(
            f"Input appears to contain personal identifiers ({summary}). "
            f"This service analyses de-identified or synthetic data; remove "
            f"the identifiers and resubmit.")


@dataclass
class PHIFinding:
    field: str
    kind: str
    # The matched text is deliberately NOT stored. Recording it here would
    # copy the identifier into logs and error payloads, which is the thing
    # detection exists to prevent.
    position: int


# Ordered most specific first: a national identifier should not be
# reported as a generic long digit run.
PATTERNS: Dict[str, re.Pattern] = {
    "national_id": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    "phone": re.compile(r"\b(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]\d{3}[ -]\d{4}\b"),
    "date_of_birth": re.compile(r"\b(?:19|20)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])\b"),
    "record_number": re.compile(r"\b\d{7,}\b"),
}

# Presidio entity types worth acting on here. Its full set includes things
# like URL and IP address that are not PHI in this context.
PRESIDIO_ENTITIES = (
    "US_SSN", "EMAIL_ADDRESS", "PHONE_NUMBER", "DATE_TIME", "PERSON",
    "LOCATION", "MEDICAL_LICENSE", "UK_NHS", "CREDIT_CARD",
)


@dataclass
class PHIScanner:
    """Scans field values for identifiers.

    `enabled=False` means no scanning happens, and callers are expected to
    say so where a reader can see it rather than leaving it implied.
    """
    enabled: bool = False
    backend: str = "patterns"
    on_detection: str = "reject"
    _analyzer: Any = field(default=None, repr=False)

    SUPPORTED_BACKENDS = ("patterns", "presidio")
    SUPPORTED_ACTIONS = ("reject", "redact")

    def __post_init__(self) -> None:
        if self.backend not in self.SUPPORTED_BACKENDS:
            raise PHIConfigurationError(
                f"Unknown PHI backend {self.backend!r}; expected one of "
                f"{self.SUPPORTED_BACKENDS}")
        if self.on_detection not in self.SUPPORTED_ACTIONS:
            raise PHIConfigurationError(
                f"Unknown on_detection {self.on_detection!r}; expected one of "
                f"{self.SUPPORTED_ACTIONS}")

        if self.enabled and self.backend == "presidio":
            try:
                from presidio_analyzer import AnalyzerEngine
            except ImportError as e:
                # Not a warning. Falling back quietly is how the config came
                # to claim a control it did not have.
                raise PHIConfigurationError(
                    f"PHI detection is configured to use Presidio, which is "
                    f"not installed ({e}). Install the `phi` extra, or set "
                    f"security.phi.backend to \"patterns\", or disable "
                    f"detection -- but do not leave it claiming to be on."
                ) from e
            self._analyzer = AnalyzerEngine()

    @property
    def describe(self) -> Dict[str, Any]:
        """What is actually being done, for the health payload."""
        if not self.enabled:
            return {
                "enabled": False,
                "note": ("No PHI detection is performed. Send de-identified "
                         "or synthetic data only."),
            }
        return {
            "enabled": True,
            "backend": self.backend,
            "on_detection": self.on_detection,
            "note": (
                "Direct identifiers matched by pattern: national id, email, "
                "phone, date of birth, long digit runs. Names and addresses "
                "are not detected."
                if self.backend == "patterns" else
                "Presidio NLP recognisers over the configured entity types."
            ),
        }

    def scan_text(self, value: str, field_name: str) -> List[PHIFinding]:
        if not self.enabled or not value:
            return []

        if self.backend == "presidio":
            return [
                PHIFinding(field_name, result.entity_type, result.start)
                for result in self._analyzer.analyze(
                    text=value, entities=list(PRESIDIO_ENTITIES), language="en")
            ]

        findings = []
        matched_spans: List[range] = []
        for kind, pattern in PATTERNS.items():
            for match in pattern.finditer(value):
                span = range(match.start(), match.end())
                # A national id also matches the digit-run rule; report the
                # specific finding once rather than the same text twice.
                if any(span.start in seen or seen.start in span
                       for seen in matched_spans):
                    continue
                matched_spans.append(span)
                findings.append(PHIFinding(field_name, kind, match.start()))
        return findings

    def scan(self, values: Dict[str, Any]) -> List[PHIFinding]:
        """Scan a mapping of field name to string, list, or dict."""
        findings: List[PHIFinding] = []
        for name, value in values.items():
            for text in _strings_in(value):
                findings.extend(self.scan_text(text, name))
        return findings

    def redact(self, value: str) -> str:
        if not self.enabled or not value:
            return value
        for pattern in PATTERNS.values():
            value = pattern.sub("[REDACTED]", value)
        return value

    def enforce(self, values: Dict[str, Any]) -> List[PHIFinding]:
        """Scan and, when configured to reject, raise on any finding."""
        findings = self.scan(values)
        if findings and self.on_detection == "reject":
            raise PHIDetected(findings)
        return findings


def _strings_in(value: Any) -> Iterable[str]:
    """Every string inside a value, including dict keys."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, nested in value.items():
            if isinstance(key, str):
                yield key
            yield from _strings_in(nested)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _strings_in(item)


def scanner_from_config(config: Optional[Dict[str, Any]]) -> PHIScanner:
    """Build a scanner from the `security.phi` block."""
    phi = (config or {}).get("security", {}).get("phi", {})
    return PHIScanner(
        enabled=bool(phi.get("enabled", False)),
        backend=phi.get("backend", "patterns"),
        on_detection=phi.get("on_detection", "reject"),
    )
