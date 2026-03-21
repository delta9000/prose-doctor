"""Tier promotion logic."""
from __future__ import annotations


def check_tier(stats: dict, revision_evidence: list[str]) -> str:
    d = abs(stats.get("cohens_d", 0))
    p = stats.get("p_value", 1.0)
    if d < 0.5 and p >= 0.01:
        return "experimental"
    if revision_evidence and len(revision_evidence) >= 2:
        total_accepted = 0
        chapters = set()
        for ev in revision_evidence:
            parts = ev.split(":")
            if len(parts) == 2 and "_accepted" in parts[1]:
                chapters.add(parts[0])
                try:
                    total_accepted += int(parts[1].split("_")[0])
                except ValueError:
                    pass
        if total_accepted >= 3 and len(chapters) >= 2:
            return "stable"
    return "validated"
