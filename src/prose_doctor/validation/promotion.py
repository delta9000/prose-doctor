"""Tier promotion logic.

Tiers reflect practical maturity:
- experimental: lens runs and measures something, but unproven for revision
- validated: good discrimination AND revision evidence showing it improves text
- stable: rock solid — extensive revision evidence across multiple chapters
"""
from __future__ import annotations


def _parse_evidence(revision_evidence: list[str]) -> tuple[int, int]:
    """Parse revision evidence strings. Returns (total_accepted, n_chapters)."""
    total_accepted = 0
    chapters: set[str] = set()
    for ev in revision_evidence:
        parts = ev.split(":")
        if len(parts) == 2 and "_accepted" in parts[1]:
            chapters.add(parts[0])
            try:
                total_accepted += int(parts[1].split("_")[0])
            except ValueError:
                pass
    return total_accepted, len(chapters)


def check_tier(stats: dict, revision_evidence: list[str]) -> str:
    """Determine a lens's tier from discrimination stats and revision evidence.

    experimental: weak stats, OR good stats but no revision evidence yet.
    validated: good discrimination (d >= 0.5 or p < 0.01) AND at least 1
               accepted revision edit — proof the lens improves text.
    stable: good discrimination AND 3+ accepted edits across 2+ chapters.
    """
    d = abs(stats.get("cohens_d", 0))
    p = stats.get("p_value", 1.0)
    has_good_stats = d >= 0.5 or p < 0.01

    if not has_good_stats:
        return "experimental"

    # Good stats but no revision evidence — stays experimental
    total_accepted, n_chapters = _parse_evidence(revision_evidence)
    if total_accepted == 0:
        return "experimental"

    # Extensive revision evidence across multiple chapters → stable
    if total_accepted >= 3 and n_chapters >= 2:
        return "stable"

    # Some revision evidence → validated
    return "validated"
