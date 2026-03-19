"""Text splitting and word counting utilities.

Handles markdown with frontmatter, plain text, and scene-break conventions.
"""

from __future__ import annotations

import re


def _has_frontmatter(text: str) -> bool:
    """Detect a header block terminated by a standalone --- line.

    Matches two formats:
    - YAML frontmatter: starts with ---, metadata, ends with ---
    - Prose header: starts with # Title, metadata lines, then ---

    In both cases, the --- line (on its own paragraph) signals "body starts here".
    """
    for block in text.split("\n\n"):
        stripped = block.strip()
        if stripped == "---":
            return True
        # Check if block ends with a --- line (YAML frontmatter closing)
        lines = stripped.split("\n")
        if len(lines) > 1 and lines[-1].strip() == "---" and lines[0].strip() == "---":
            return True
    return False


def _skip_frontmatter(text: str) -> str:
    """Return text after the header block (terminated by standalone ---)."""
    if not _has_frontmatter(text):
        return text
    # Find the last --- line that's part of the header
    blocks = text.split("\n\n")
    body_parts = []
    in_header = True
    for block in blocks:
        stripped = block.strip()
        if not stripped:
            continue
        if in_header:
            if stripped == "---":
                in_header = False
                continue
            lines = stripped.split("\n")
            if (
                len(lines) > 1
                and lines[0].strip() == "---"
                and lines[-1].strip() == "---"
            ):
                in_header = False
                continue
            continue
        body_parts.append(block)
    return "\n\n".join(body_parts) if body_parts else text


def split_paragraphs(text: str) -> list[str]:
    """Split text into body paragraphs, skipping frontmatter and scene breaks.

    Auto-detects format:
    - Markdown with --- frontmatter: skips header, treats subsequent --- as scene breaks
    - Plain text: splits on blank lines
    - Headings (# lines) are always skipped
    """
    if _has_frontmatter(text):
        # Frontmatter format: skip header, then split body
        paragraphs = []
        in_header = True
        for block in text.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if in_header:
                if block == "---":
                    # Standalone --- line terminates header
                    in_header = False
                    continue
                # YAML frontmatter: block starts and ends with ---
                lines = block.split("\n")
                if (
                    len(lines) > 1
                    and lines[0].strip() == "---"
                    and lines[-1].strip() == "---"
                ):
                    in_header = False
                    continue
                continue
            if block == "---" or block.startswith("#"):
                continue
            paragraphs.append(block)
        return paragraphs
    else:
        # Plain text: just split on blank lines
        return [
            block.strip()
            for block in text.split("\n\n")
            if block.strip() and not block.strip().startswith("#")
        ]


def split_paragraphs_with_breaks(text: str) -> list[str | None]:
    """Split text into paragraphs, marking scene breaks as None.

    Returns a list where each element is either a paragraph string or None
    (representing a scene break). Preserves break positions for context tracking.
    """
    if _has_frontmatter(text):
        items: list[str | None] = []
        in_header = True
        for block in text.split("\n\n"):
            block = block.strip()
            if not block:
                continue
            if in_header:
                if block == "---":
                    in_header = False
                    continue
                lines = block.split("\n")
                if (
                    len(lines) > 1
                    and lines[0].strip() == "---"
                    and lines[-1].strip() == "---"
                ):
                    in_header = False
                    continue
                continue
            if block.startswith("#"):
                continue
            if block == "---":
                items.append(None)
            else:
                items.append(block)
        return items
    else:
        return [
            block.strip()
            for block in text.split("\n\n")
            if block.strip() and not block.strip().startswith("#")
        ]


def count_words(text: str) -> int:
    """Count words in body text, skipping frontmatter."""
    body = _skip_frontmatter(text)
    return len(body.split())


def count_words_in_paragraphs(paragraphs: list[str]) -> int:
    """Count total words across a list of paragraphs."""
    return sum(len(p.split()) for p in paragraphs)


def is_dialogue_line(line: str) -> bool:
    """Heuristic: line contains quoted speech."""
    return '"' in line or "\u201c" in line or "\u201d" in line
