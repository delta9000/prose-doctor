"""Density budget system: catches patterns that are fine individually but
become tiresome through repetition."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from prose_doctor.patterns.rules import RulePattern, build_rule_patterns, check_rules
from prose_doctor.text import split_paragraphs


@dataclass
class DensityHit:
    """A single detected density pattern instance."""

    pattern: str
    text: str
    paragraph_index: int
    severity: str  # "info", "warn", "over"


@dataclass
class DensityReport:
    """Chapter-level density analysis."""

    filename: str
    word_count: int
    hits: list[DensityHit] = field(default_factory=list)
    pattern_counts: dict[str, int] = field(default_factory=dict)
    pattern_budgets: dict[str, int] = field(default_factory=dict)
    over_budget: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"{self.filename} ({self.word_count:,} words)"]
        over_items = [
            (p, c)
            for p, c in sorted(self.pattern_counts.items())
            if c > self.pattern_budgets.get(p, 999)
        ]
        ok_items = [
            (p, c)
            for p, c in sorted(self.pattern_counts.items())
            if c <= self.pattern_budgets.get(p, 999)
        ]
        for pat, count in over_items:
            budget = self.pattern_budgets.get(pat, 999)
            lines.append(f"  {pat:30s} {count:3d} / {budget:3d} X ({count - budget:+d})")
        for pat, count in ok_items:
            budget = self.pattern_budgets.get(pat, 999)
            lines.append(f"  {pat:30s} {count:3d} / {budget:3d}")
        return "\n".join(lines)


# Budget table: max instances per 2000 words (scaled by chapter length)
BUDGETS_PER_2K: dict[str, int] = {
    "tricolon": 2,
    "explanatory_doubling": 1,
    "over_resolution": 1,
    "cost_recovery": 2,
    "role_quality_label": 1,
    "patient_nonperson": 1,
    "deep_resonant": 0,
    "phantom_sensation": 1,
    "prepositional_stacking": 3,
}


def _budget_for_chapter(pattern: str, word_count: int) -> int:
    """Scale the per-2K budget to the actual chapter length."""
    if pattern.startswith("word:") or pattern.startswith("phrase:"):
        return 999
    per_2k = BUDGETS_PER_2K.get(pattern, 2)
    return max(1, int(per_2k * (word_count / 2000)))


class DensityAnalyzer:
    """Density analysis for prose patterns (regex-only, no spaCy required)."""

    def __init__(
        self,
        character_names: list[str] | None = None,
        budget_overrides: dict[str, int] | None = None,
    ):
        self._patterns = build_rule_patterns(character_names)
        self._density_rules = {
            "role_quality_label", "patient_nonperson",
            "deep_resonant", "phantom_sensation",
        }
        if budget_overrides:
            BUDGETS_PER_2K.update(budget_overrides)

    def _detect_word_density(
        self, paragraphs: list[str], word_count: int
    ) -> list[DensityHit]:
        """Flag words/phrases that appear too frequently."""
        hits = []
        full_text = " ".join(paragraphs).lower()

        tracked_words = {
            "precise": 0.8, "precision": 0.6, "particular": 0.6,
            "steady": 1.0, "layered": 1.0, "clinical": 0.4,
            "clean": 1.0, "flat": 1.0,
            "pulsed": 1.0, "throbbed": 0.6, "thrummed": 0.4,
            "vast": 0.6, "ancient": 0.6, "profound": 0.3,
        }

        tracked_phrases = {
            "the shape of": 1.0,
            "the weight of": 1.0,
            "something in": 1.0,
        }

        for word, threshold in tracked_words.items():
            count = len(re.findall(rf'\b{word}\b', full_text))
            budget = max(2, int(threshold * word_count / 1000))
            if count > budget:
                hits.append(DensityHit(
                    f"word:{word}",
                    f'"{word}" appears {count}x (budget: {budget} for {word_count} words)',
                    -1, "over",
                ))

        for phrase, threshold in tracked_phrases.items():
            count = len(re.findall(re.escape(phrase), full_text))
            budget = max(1, int(threshold * word_count / 1000))
            if count > budget:
                hits.append(DensityHit(
                    f"phrase:{phrase}",
                    f'"{phrase}" appears {count}x (budget: {budget} for {word_count} words)',
                    -1, "over",
                ))

        return hits

    def _detect_over_resolution(self, paragraphs: list[str]) -> list[DensityHit]:
        """Detect complete images followed by redundant trailing simile."""
        hits = []
        like_pat = re.compile(
            r'([^.]{40,}?),?\s+like\s+([^.]{5,40})\.'
        )

        sensory_pat = re.compile(
            r'\b(?:pressed|settled|cold|warm|hot|sharp|dull|heavy|'
            r'tight|smooth|rough|hard|soft|wet|dry|dark|bright|'
            r'blood|bone|skin|metal|glass|stone|wood|iron|salt|'
            r'copper|breath|pulse|ache|sting|burn|throb)\b',
            re.IGNORECASE,
        )

        for pi, para in enumerate(paragraphs):
            for m in like_pat.finditer(para):
                preceding = m.group(1)
                sensory_words = sensory_pat.findall(preceding)
                if len(sensory_words) >= 2:
                    text = f"...{preceding[-60:]} like {m.group(2)}"
                    hits.append(DensityHit("over_resolution", text[:120], pi, "info"))

        return hits

    def _detect_cost_recovery(self, paragraphs: list[str]) -> list[DensityHit]:
        """Detect ability -> symptom -> wipe -> continue formula."""
        hits = []
        cost_pat = re.compile(
            r'\b(?:iron taste|tasted (?:of )?iron|copper taste|tasted (?:of )?copper|'
            r'salt on (?:her|his) (?:tongue|lips)|nosebleed|blood (?:from|on) (?:his|her) '
            r'(?:nose|lip|nostril)|wiped (?:the )?blood|the taste faded|'
            r'the (?:iron|copper|salt) faded)\b',
            re.IGNORECASE,
        )
        recovery_pat = re.compile(
            r'\b(?:faded|cleared|subsided|passed|wiped|eased|retreated|'
            r'the (?:taste|pain|pressure|ache) (?:faded|cleared|eased|subsided))\b',
            re.IGNORECASE,
        )

        for pi, para in enumerate(paragraphs):
            if cost_pat.search(para) and recovery_pat.search(para):
                short = para[:100].replace("\n", " ")
                hits.append(DensityHit("cost_recovery", short, pi, "info"))

        return hits

    def analyze(self, text: str, filename: str = "<unknown>") -> DensityReport:
        """Run all density checks on text and return a report."""
        paragraphs = split_paragraphs(text)
        word_count = sum(len(p.split()) for p in paragraphs)

        all_hits: list[DensityHit] = []
        all_hits.extend(self._detect_over_resolution(paragraphs))
        all_hits.extend(self._detect_cost_recovery(paragraphs))
        all_hits.extend(self._detect_word_density(paragraphs, word_count))

        # Run density-category rule patterns
        for pi, para in enumerate(paragraphs):
            for m in check_rules(para, self._patterns):
                if m["pattern_name"] in self._density_rules:
                    short = para[:100].replace("\n", " ")
                    all_hits.append(DensityHit(m["pattern_name"], short, pi, "info"))

        # Compute counts and budgets
        pattern_counts: dict[str, int] = {}
        for h in all_hits:
            pattern_counts[h.pattern] = pattern_counts.get(h.pattern, 0) + 1

        pattern_budgets = {
            pat: _budget_for_chapter(pat, word_count)
            for pat in pattern_counts
        }

        over_budget: dict[str, int] = {}
        for pat, count in pattern_counts.items():
            budget = pattern_budgets[pat]
            if pat.startswith("word:") or pat.startswith("phrase:"):
                over_budget[pat] = count
            elif count > budget:
                over_budget[pat] = count - budget

        # Tag severity
        for h in all_hits:
            budget = pattern_budgets.get(h.pattern, 999)
            running = sum(
                1 for h2 in all_hits
                if h2.pattern == h.pattern and h2.paragraph_index <= h.paragraph_index
            )
            if running > budget:
                h.severity = "over"
            elif running == budget:
                h.severity = "warn"

        return DensityReport(
            filename=filename,
            word_count=word_count,
            hits=all_hits,
            pattern_counts=pattern_counts,
            pattern_budgets=pattern_budgets,
            over_budget=over_budget,
        )
