"""JSON serialization for prose-doctor reports."""

from __future__ import annotations

import json

from prose_doctor.analyzers.doctor import ChapterHealth


def reports_to_json(reports: list[ChapterHealth]) -> str:
    """Serialize a list of ChapterHealth reports to JSON."""
    total_words = sum(r.word_count for r in reports)
    total_issues = sum(r.total_issues for r in reports)

    data = {
        "summary": {
            "chapters": len(reports),
            "total_words": total_words,
            "total_issues": total_issues,
        },
        "chapters": [r.to_dict() for r in reports],
    }
    return json.dumps(data, indent=2)
