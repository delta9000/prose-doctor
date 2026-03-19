"""Proof scanner: regex pattern bank with HIGH/MEDIUM severity findings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from prose_doctor.patterns.rules import RulePattern, build_rule_patterns
from prose_doctor.text import is_dialogue_line


@dataclass
class Finding:
    """A single proof scan finding."""

    line: int
    category: str
    match: str
    context: str
    severity: str  # "high" or "medium"


class ProofScanner:
    """Scans text for prose patterns at line level."""

    def __init__(self, character_names: list[str] | None = None):
        self._patterns = build_rule_patterns(character_names)
        # Separate by severity for scanning
        self._high = [p for p in self._patterns if p.severity == "high"]
        self._medium = [p for p in self._patterns if p.severity == "medium"]

    def scan(self, text: str) -> list[Finding]:
        """Scan text and return all findings with line numbers."""
        lines = text.splitlines()
        findings: list[Finding] = []

        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                continue

            dialogue = is_dialogue_line(line)

            # High severity patterns
            for rule in self._high:
                for m in rule.regex.finditer(line):
                    findings.append(Finding(
                        line=lineno,
                        category=rule.name,
                        match=m.group()[:80],
                        context=stripped[:120],
                        severity="high",
                    ))

            # Medium severity patterns (some skip dialogue)
            for rule in self._medium:
                # Skip some patterns in dialogue
                if dialogue and rule.name in (
                    "wasnt_it_was", "not_x_but_y", "weight_of_abstraction",
                    "something_in_pronoun", "seemed_to",
                ):
                    continue

                for m in rule.regex.finditer(line):
                    findings.append(Finding(
                        line=lineno,
                        category=rule.name,
                        match=m.group()[:80],
                        context=stripped[:120],
                        severity="medium",
                    ))

        return findings

    def scan_file(self, path: Path) -> list[Finding]:
        """Scan a file and return findings."""
        return self.scan(path.read_text())
