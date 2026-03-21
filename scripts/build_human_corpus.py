#!/usr/bin/env python3
"""Download and prepare the human baseline corpus.

Sources:
  - Project Gutenberg (public domain)
  - Creative Commons licensed novels (various CC licenses)

Outputs chapter-split markdown files to corpus/human/{source}/{author}_{title}_ch{N}.md
"""
from __future__ import annotations

import json
import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# Force IPv4 — Gutenberg load-balances to IPv6 which may be unreachable
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_getaddrinfo

CORPUS_DIR = Path("corpus/human")

# --- Gutenberg works ---
# Format: (gutenberg_id, author_slug, title_slug, genre)
GUTENBERG_WORKS = [
    (2852,  "doyle",        "hound_of_the_baskervilles",  "detective"),
    # Mystery / Thriller
    (58866, "christie",     "murder_on_the_links",        "mystery"),
    (558,   "buchan",       "thirty_nine_steps",          "thriller"),
    (1155,  "poe",          "works_of_poe_v2",            "gothic_horror"),
    # Sci-fi / Speculative
    (36,    "wells",        "war_of_the_worlds",          "sci-fi"),
    (35,    "wells",        "time_machine",               "sci-fi"),
    # Adventure / Western
    (215,   "london",       "call_of_the_wild",           "adventure"),
    (1300,  "grey",         "riders_of_the_purple_sage",  "western"),
    # Horror / Gothic
    (345,   "stoker",       "dracula",                    "gothic"),
    (84,    "shelley",      "frankenstein",               "gothic_sf"),
    # Romance / Social
    (1342,  "austen",       "pride_and_prejudice",        "romance"),
    (541,   "wharton",      "age_of_innocence",           "social"),
    # Psychological / Literary
    (2554,  "dostoevsky",   "crime_and_punishment",       "psychological"),
    (219,   "conrad",       "heart_of_darkness",          "literary"),
    (98,    "dickens",      "tale_of_two_cities",         "historical"),
    (64317, "fitzgerald",   "great_gatsby",               "literary"),
    # Satire / Humor
    (76,    "twain",        "huckleberry_finn",           "satire"),
    # Russian / Translation
    (1399,  "tolstoy",      "anna_karenina",              "literary"),
    # French / Adventure
    (164,   "verne",        "twenty_thousand_leagues",    "adventure_sf"),
    # American naturalism
    (1098,  "crane",        "red_badge_of_courage",       "war"),
]

# --- CC-licensed works ---
# Format: (url, author_slug, title_slug, genre, license)
CC_WORKS = [
    # Doctorow — direct .txt links
    ("http://craphound.com/littlebrother/Cory_Doctorow_-_Little_Brother.txt",
     "doctorow", "little_brother", "ya_sf", "CC BY-NC-SA"),
    ("http://craphound.com/makers/Cory_Doctorow_-_Makers.txt",
     "doctorow", "makers", "near_future", "CC BY-NC-SA"),
    ("http://craphound.com/down/Cory_Doctorow_-_Down_and_Out_in_the_Magic_Kingdom.txt",
     "doctorow", "down_and_out", "sci-fi", "CC BY-NC-SA-ND"),
    ("http://craphound.com/est/Cory_Doctorow_-_Eastern_Standard_Tribe.txt",
     "doctorow", "eastern_standard_tribe", "sci-fi", "CC BY-NC-SA-ND"),
    ("http://craphound.com/ftw/Cory_Doctorow_-_For_the_Win.txt",
     "doctorow", "for_the_win", "ya_thriller", "CC BY-NC-SA"),
    # Watts — full HTML (sections: Prologue, Theseus, Rorschach, Charybdis)
    ("https://rifters.com/real/Blindsight.htm",
     "watts", "blindsight", "hard_sf", "CC BY-NC-SA"),
    # Stross — full HTML (needs SSL bypass for antipope.org cert)
    ("https://www.antipope.org/charlie/blog-static/fiction/accelerando/accelerando.html",
     "stross", "accelerando", "sci-fi", "CC BY-NC-ND"),
]


def fetch_gutenberg_text(book_id: int) -> str | None:
    """Fetch plain text from Project Gutenberg."""
    # Try the plain text URL patterns
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    for url in urls:
        for attempt in range(3):
            try:
                req = Request(url, headers={"User-Agent": "prose-doctor-corpus-builder/1.0"})
                with urlopen(req, timeout=30) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                    if len(text) > 1000:
                        return text
            except (HTTPError, URLError, OSError) as e:
                if attempt < 2:
                    time.sleep(3)
                    continue
            time.sleep(2)  # be polite
    return None


def strip_gutenberg_header_footer(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    # Find start of actual text
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Skip to next blank line after marker
            newline = text.find("\n\n", idx)
            if newline != -1:
                text = text[newline + 2:]
            break

    # Find end of actual text
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    return text.strip()


def split_into_chapters(text: str) -> list[tuple[str, str]]:
    """Split text into chapters. Returns list of (chapter_title, chapter_text)."""
    # Try common chapter heading patterns
    patterns = [
        re.compile(r'^(CHAPTER\s+[IVXLC\d]+[.\s:—–-].*?)$', re.MULTILINE | re.IGNORECASE),
        re.compile(r'^(Chapter\s+[IVXLC\d]+[.\s:—–-].*?)$', re.MULTILINE),
        re.compile(r'^(BOOK\s+[IVXLC\d]+[.\s:—–-].*?)$', re.MULTILINE | re.IGNORECASE),
        re.compile(r'^([IVXLC]+\.\s+.+)$', re.MULTILINE),  # "I. The Beginning"
        re.compile(r'^(\d+\.\s+.+)$', re.MULTILINE),  # "1. The Beginning"
    ]

    for pattern in patterns:
        matches = list(pattern.finditer(text))
        if len(matches) >= 3:  # need at least 3 chapters to be confident
            chapters = []
            for i, m in enumerate(matches):
                title = m.group(1).strip()
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                chapter_text = text[start:end].strip()
                if len(chapter_text.split()) >= 500:  # skip very short chapters
                    chapters.append((title, chapter_text))
            if chapters:
                return chapters

    # Fallback: split by double newlines into ~3000 word chunks
    paragraphs = text.split("\n\n")
    chapters = []
    current = []
    current_words = 0
    for para in paragraphs:
        words = len(para.split())
        current.append(para)
        current_words += words
        if current_words >= 3000:
            chapters.append((f"Section {len(chapters) + 1}", "\n\n".join(current)))
            current = []
            current_words = 0
    if current and current_words >= 500:
        chapters.append((f"Section {len(chapters) + 1}", "\n\n".join(current)))

    return chapters


def save_chapters(
    chapters: list[tuple[str, str]],
    author: str,
    title: str,
    source: str,
    max_chapters: int = 15,
) -> int:
    """Save chapters to corpus/human/{source}/. Returns count saved."""
    out_dir = CORPUS_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, (ch_title, ch_text) in enumerate(chapters[:max_chapters]):
        filename = f"{author}_{title}_ch{i:02d}.md"
        path = out_dir / filename
        path.write_text(ch_text)
        saved += 1

    return saved


def build_gutenberg_corpus() -> dict:
    """Download and process all Gutenberg works."""
    stats = {}
    for book_id, author, title, genre in GUTENBERG_WORKS:
        key = f"{author}/{title}"
        print(f"  Fetching Gutenberg #{book_id}: {key}...", end=" ", flush=True)

        text = fetch_gutenberg_text(book_id)
        if text is None:
            print("FAILED (not found)")
            stats[key] = {"status": "failed", "reason": "not found"}
            continue

        text = strip_gutenberg_header_footer(text)
        chapters = split_into_chapters(text)
        if not chapters:
            print(f"FAILED (no chapters found, {len(text)} chars)")
            stats[key] = {"status": "failed", "reason": "no chapters"}
            continue

        saved = save_chapters(chapters, author, title, "gutenberg")
        print(f"OK ({saved} chapters, {genre})")
        stats[key] = {
            "status": "ok",
            "chapters": saved,
            "genre": genre,
            "source": "gutenberg",
            "license": "public_domain",
            "gutenberg_id": book_id,
        }
        time.sleep(2)  # be polite to Gutenberg

    return stats


def build_cc_corpus() -> dict:
    """Download and process CC-licensed works."""
    stats = {}
    for url, author, title, genre, license_type in CC_WORKS:
        key = f"{author}/{title}"
        print(f"  Fetching CC: {key}...", end=" ", flush=True)

        try:
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = Request(url, headers={"User-Agent": "prose-doctor-corpus-builder/1.0"})
            with urlopen(req, timeout=30, context=ctx) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read()

                if "pdf" in content_type or url.endswith(".pdf"):
                    print("SKIPPED (PDF — needs manual processing)")
                    stats[key] = {"status": "skipped", "reason": "pdf"}
                    continue

                text = raw.decode("utf-8", errors="replace")
        except Exception as e:
            print(f"FAILED ({e})")
            stats[key] = {"status": "failed", "reason": str(e)}
            continue

        # Strip HTML if needed
        if "<html" in text.lower()[:500] or "<body" in text.lower()[:500]:
            # Basic HTML strip — good enough for chapter splitting
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', '\n', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()

        chapters = split_into_chapters(text)
        if not chapters:
            print(f"FAILED (no chapters found, {len(text)} chars)")
            stats[key] = {"status": "failed", "reason": "no chapters"}
            continue

        saved = save_chapters(chapters, author, title, "cc")
        print(f"OK ({saved} chapters, {genre}, {license_type})")
        stats[key] = {
            "status": "ok",
            "chapters": saved,
            "genre": genre,
            "source": "cc",
            "license": license_type,
        }
        time.sleep(2)

    return stats


def main():
    print("Building human baseline corpus...\n")

    print("=== Project Gutenberg (public domain) ===")
    gutenberg_stats = build_gutenberg_corpus()

    print("\n=== Creative Commons ===")
    cc_stats = build_cc_corpus()

    # Write manifest
    all_stats = {**gutenberg_stats, **cc_stats}
    manifest_path = CORPUS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(all_stats, indent=2))

    # Summary
    ok = sum(1 for v in all_stats.values() if v.get("status") == "ok")
    total_chapters = sum(v.get("chapters", 0) for v in all_stats.values())
    failed = sum(1 for v in all_stats.values() if v.get("status") == "failed")
    skipped = sum(1 for v in all_stats.values() if v.get("status") == "skipped")

    print(f"\n=== Summary ===")
    print(f"  Works downloaded: {ok}")
    print(f"  Total chapters:   {total_chapters}")
    print(f"  Failed:           {failed}")
    print(f"  Skipped (PDF):    {skipped}")
    print(f"  Manifest:         {manifest_path}")


if __name__ == "__main__":
    main()
