"""Human-readable report formatting."""

from __future__ import annotations

from collections import Counter

from prose_doctor.analyzers.doctor import ChapterHealth


def format_chapter_report(health: ChapterHealth) -> str:
    """Format a ChapterHealth as a human-readable report."""
    lines = [
        f"\n{'=' * 70}",
        f"  {health.filename} ({health.word_count:,} words) -- {health.total_issues} issues",
        f"{'=' * 70}",
    ]

    # ML results (if available)
    if health.perplexity:
        p = health.perplexity
        lines.append(
            f"\n  PERPLEXITY: mean={p['mean_ppl']}, {p['pct_below_55']}% smooth"
        )
        for s in p.get("smoothest_paragraphs", [])[:3]:
            lines.append(
                f"      ppl={s['perplexity']:>5.1f} [{s['index']:>3}] {s['text'][:70]}"
            )

    if health.foregrounding:
        f = health.foregrounding
        lines.append(
            f"\n  FOREGROUNDING: index={f['index']} | "
            f"allit={f['alliteration_per_1k']} inv={f['inversion_pct']}% "
            f"cv={f['sentence_length_cv']} frag={f['fragment_pct']}%"
        )
        lines.append(f"    Weakest: {f['weakest_axis']} -> {f['prescription']}")

    if health.emotion:
        e = health.emotion
        flat_tag = " <- FLAT" if e.get("flat") else ""
        arc = e.get("arc", "?")
        lines.append(f"\n  EMOTION ARC: {arc} std={e.get('std', 0)}{flat_tag}")

    if health.psychic_distance:
        pd = health.psychic_distance
        lines.append(
            f"\n  PSYCHIC DISTANCE: mean={pd['mean_distance']} ({pd['label']}) | "
            f"std={pd['std_distance']} | {pd['zoom_jumps']} zoom jumps"
        )

    if health.info_contour:
        ic = health.info_contour
        lines.append(
            f"\n  INFO CONTOUR: {ic['label']} | "
            f"cycle=~{ic['dominant_period']}sent (~{ic['dominant_period_words']}w) | "
            f"rhythmicity={ic['rhythmicity']} | "
            f"{ic['flatlines']} flatlines, {ic['spikes']} spikes"
        )

    if health.sensory:
        s = health.sensory
        scores = s["scores"]
        max_s = max(scores.values()) or 1
        lines.append(
            f"\n  SENSORY: dominant={s['dominant']} | "
            f"weakest={s['weakest']} | balance={s['balance']}"
        )
        for mod, val in scores.items():
            bar_w = 20
            filled = int(val / max_s * bar_w)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            lines.append(f"    {mod:<15} [{bar}] {val:.3f}")
        if s.get("prescription"):
            lines.append(f"    Rx: {s['prescription']}")

    # Vocabulary
    if health.vocabulary_crutches:
        lines.append("\n  VOCABULARY CRUTCHES:")
        for c in health.vocabulary_crutches[:12]:
            locs = ",".join(str(loc) for loc in c["locations"][:6])
            more = "..." if len(c["locations"]) > 6 else ""
            lines.append(
                f"    {c['word']:<15s} {c['count']:>3}x "
                f"(budget {c['budget']}, +{c['excess']})  "
                f"lines: {locs}{more}"
            )

    # Pattern hits
    if health.pattern_hits:
        by_severity: dict[str, list[dict]] = {"high": [], "medium": []}
        for h in health.pattern_hits:
            sev = h.get("severity", "medium")
            by_severity.setdefault(sev, []).append(h)

        if by_severity.get("high"):
            lines.append(f"\n  HIGH SEVERITY ({len(by_severity['high'])} hits):")
            by_rule = Counter(h["pattern"] for h in by_severity["high"])
            for rule, n in by_rule.most_common():
                lines.append(f"    {rule:<30s} {n:>3}")

        if by_severity.get("medium"):
            lines.append(f"\n  MEDIUM SEVERITY ({len(by_severity['medium'])} hits):")
            by_rule = Counter(h["pattern"] for h in by_severity["medium"])
            for rule, n in by_rule.most_common():
                lines.append(f"    {rule:<30s} {n:>3}")

    # Density
    if health.density_over_budget:
        lines.append("\n  DENSITY OVER BUDGET:")
        for pat, over in sorted(
            health.density_over_budget.items(), key=lambda x: -x[1]
        ):
            lines.append(f"    {pat:<30s} +{over}")

    if health.colon_lists:
        lines.append(f"\n  COLON-LISTS ({len(health.colon_lists)}):")
        for h in health.colon_lists:
            lines.append(f"    line {h['line']}: {h['text'][:80]}")

    return "\n".join(lines)


def format_summary(reports: list[ChapterHealth]) -> str:
    """Format a multi-chapter summary."""
    lines = [
        f"\n{'=' * 70}",
        "SUMMARY",
        f"{'=' * 70}",
    ]

    total_words = sum(r.word_count for r in reports)
    total_issues = sum(r.total_issues for r in reports)
    lines.append(f"  {len(reports)} chapters, {total_words:,} words, {total_issues} total issues")

    # Aggregate vocabulary crutches
    global_words: Counter = Counter()
    for r in reports:
        for c in r.vocabulary_crutches:
            global_words[c["word"]] += c["excess"]

    if global_words:
        lines.append("\n  TOP VOCABULARY CRUTCHES (total excess):")
        for word, excess in global_words.most_common(20):
            lines.append(f"    {word:<15s} +{excess}")

    # Aggregate patterns
    global_patterns: Counter = Counter()
    for r in reports:
        for h in r.pattern_hits:
            global_patterns[h["pattern"]] += 1

    if global_patterns:
        lines.append("\n  PATTERN HITS (total):")
        for pat, n in global_patterns.most_common():
            lines.append(f"    {pat:<30s} {n:>4}")

    # Worst chapters
    ranked = sorted(reports, key=lambda r: -r.total_issues)
    worst = [(r.filename, r.total_issues) for r in ranked[:5] if r.total_issues > 0]
    if worst:
        lines.append("\n  WORST CHAPTERS:")
        for name, issues in worst:
            lines.append(f"    {name:<42s} {issues} issues")

    return "\n".join(lines)
