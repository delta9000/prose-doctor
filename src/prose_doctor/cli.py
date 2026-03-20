"""CLI dispatcher for prose-doctor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prose_doctor.analyzers.doctor import ChapterHealth, diagnose
from prose_doctor.config import ProjectConfig
from prose_doctor.output.json_output import reports_to_json
from prose_doctor.output.table import format_chapter_report, format_summary


def _discover_files(paths: list[str]) -> list[Path]:
    """Expand file arguments: accept files, directories, and globs."""
    result: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(sorted(path.rglob("*.md")))
            result.extend(sorted(path.rglob("*.txt")))
        else:
            # Try glob expansion
            import glob

            expanded = glob.glob(p, recursive=True)
            result.extend(Path(e) for e in sorted(expanded) if Path(e).is_file())
    return result


def cmd_scan(args: argparse.Namespace) -> None:
    """Run the scan command."""
    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    config = ProjectConfig.load(files[0].parent)

    reports: list[ChapterHealth] = []
    for f in files:
        text = f.read_text()
        report = diagnose(text, filename=f.name, config=config)

        # Deep mode: run ML analyzers
        if args.deep:
            try:
                from prose_doctor.ml import require_ml

                require_ml()
                from prose_doctor.ml.models import ModelManager

                mm = ModelManager()

                # Classifier with density budgets — only report classes
                # that exceed a per-1000-word threshold. Individual hits
                # are candidates; accumulation makes them findings.
                try:
                    from collections import Counter
                    from prose_doctor.ml.slop_scorer import SlopScorer

                    scorer = SlopScorer(model_manager=mm, config=config)
                    stats = scorer.chapter_stats(text)
                    word_count = report.word_count or 1

                    # Count hits per class
                    class_hits: dict[str, list[dict]] = {}
                    for s in stats.get("scored", []):
                        if s["slop_prob"] > 0.5 and s.get("class_name", "clean") != "clean":
                            class_hits.setdefault(s["class_name"], []).append(s)

                    # Density budgets: hits per 1000 words before reporting
                    ML_DENSITY_BUDGETS = {
                        "thesis": 2,
                        "emotion": 2,
                        "dead_figure": 2,
                        "standalone": 3,
                        "narrator_gloss": 2,
                        "forbidden": 1,  # strict — any forbidden word cluster matters
                        "padding": 4,    # lenient — some padding is normal
                    }

                    for cls_name, hits in class_hits.items():
                        budget = ML_DENSITY_BUDGETS.get(cls_name, 2)
                        density = len(hits) / word_count * 1000
                        if density > budget:
                            # Over budget — report all hits
                            report.pattern_hits.extend(
                                {
                                    "pattern": f"ml:{cls_name}",
                                    "line": 0,
                                    "match": s["text"][:80],
                                    "severity": "ml",
                                }
                                for s in hits
                            )
                except Exception as e:
                    print(
                        f"  Classifier skipped: {e}",
                        file=sys.stderr,
                    )

                # Foregrounding (always available with ML deps)
                from prose_doctor.ml.foregrounding import score_chapter

                fg = score_chapter(text, f.name, mm)
                report.foregrounding = {
                    "index": round(fg.index, 2),
                    "alliteration_per_1k": round(fg.alliteration, 1),
                    "inversion_pct": round(fg.inversion_pct, 1),
                    "sentence_length_cv": round(fg.sl_cv, 2),
                    "fragment_pct": round(fg.fragment_pct, 1),
                    "weakest_axis": fg.weakest_axis,
                    "prescription": fg.prescription,
                }

                # Perplexity
                from prose_doctor.ml.perplexity import PerplexityScorer

                ppl = PerplexityScorer(model_manager=mm)
                report.perplexity = ppl.score_chapter(text, filename=f.name)

                # Emotion arc
                from prose_doctor.ml.emotion import EmotionArcAnalyzer

                ea = EmotionArcAnalyzer(model_manager=mm)
                emo = ea.analyze_chapter(text, filename=f.name)
                report.emotion = {
                    "flat": emo["flat"],
                    "arc": emo.get("arc", "?"),
                    "std": emo.get("stats", {}).get("std", 0),
                    "dynamic_range": emo.get("stats", {}).get("dynamic_range", 0),
                }

                # Psychic distance
                from prose_doctor.ml.psychic_distance import analyze_chapter as pd_analyze

                pd = pd_analyze(text, f.name, mm)
                report.psychic_distance = {
                    "mean_distance": pd.mean_distance,
                    "std_distance": pd.std_distance,
                    "label": pd.label,
                    "zoom_jumps": len(pd.zoom_jumps),
                    "paragraph_means": [round(m, 3) for m in pd.paragraph_means],
                }

                # Information contour
                from prose_doctor.ml.info_contour import analyze_chapter as ic_analyze

                ic = ic_analyze(text, f.name, mm)
                report.info_contour = {
                    "mean_surprisal": ic.mean_surprisal,
                    "cv_surprisal": ic.cv_surprisal,
                    "label": ic.label,
                    "dominant_period": ic.dominant_period,
                    "dominant_period_words": ic.dominant_period_words,
                    "rhythmicity": ic.rhythmicity,
                    "flatlines": len(ic.flatlines),
                    "spikes": len(ic.spikes),
                }

                # Sensory profiler
                from prose_doctor.ml.sensory import profile_chapter

                sp = profile_chapter(text, f.name, mm)
                report.sensory = {
                    "dominant": sp.dominant_modality,
                    "weakest": sp.weakest_modality,
                    "balance": round(sp.balance_ratio, 3),
                    "scores": sp.scores,
                    "deserts": len(sp.deserts),
                    "prescription": sp.prescription,
                }

            except ImportError as e:
                print(f"ML features unavailable: {e}", file=sys.stderr)
                args.deep = False

        reports.append(report)

    if args.json:
        print(reports_to_json(reports))
    else:
        for report in reports:
            print(format_chapter_report(report))
        if len(reports) > 1:
            print(format_summary(reports))


def cmd_init(args: argparse.Namespace) -> None:
    """Generate a .prose-doctor.toml template."""
    target = Path.cwd() / ".prose-doctor.toml"
    if target.exists():
        print(f"{target} already exists.", file=sys.stderr)
        sys.exit(1)
    target.write_text(ProjectConfig.default_template())
    print(f"Created {target}")


def cmd_index(args: argparse.Namespace) -> None:
    """Run foregrounding index analysis."""
    try:
        from prose_doctor.ml import require_ml

        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    from prose_doctor.ml.foregrounding import score_chapter
    from prose_doctor.ml.models import ModelManager

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    mm = ModelManager()
    scores = []
    for f in files:
        text = f.read_text()
        sc = score_chapter(text, f.name, mm)
        scores.append(sc)

    scores.sort(key=lambda s: s.index)

    header = (
        f"{'Chapter':<42} {'Words':>5} {'Allit':>5} {'Inv%':>5} "
        f"{'SLCV':>5} {'Frg%':>5} {'INDEX':>6}  Weakest"
    )
    print(header)
    print("-" * len(header))
    for sc in scores:
        print(
            f"{sc.filename:<42} {sc.word_count:>5} {sc.alliteration:>5.1f} "
            f"{sc.inversion_pct:>4.1f}% {sc.sl_cv:>5.2f} {sc.fragment_pct:>4.1f}% "
            f"{sc.index:>6.2f}  {sc.weakest_axis}"
        )


def cmd_twins(args: argparse.Namespace) -> None:
    """Run twin-finder analysis."""
    try:
        from prose_doctor.ml import require_ml

        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    from prose_doctor.ml.models import ModelManager
    from prose_doctor.ml.twins import find_twins

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    mm = ModelManager()
    twins = find_twins(files, mm, max_results=getattr(args, "top", 20))

    if not twins:
        print("No twins found (too few paragraphs or insufficient texture variation).")
        return

    for i, tw in enumerate(twins, 1):
        print(f"--- Match {i} ---")
        print(f"FLAT  [{tw.flat_file}:{tw.flat_idx}] texture={tw.flat_texture:.3f}")
        print(f"  {tw.flat_text[:160]}")
        print(
            f"TWIN  [{tw.twin_file}:{tw.twin_idx}] texture={tw.twin_texture:.3f} "
            f"sim={tw.topical_similarity:.2f}"
        )
        print(f"  {tw.twin_text[:160]}")
        print()


def cmd_distance(args: argparse.Namespace) -> None:
    """Run psychic distance analysis."""
    try:
        from prose_doctor.ml import require_ml

        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    from prose_doctor.ml.models import ModelManager
    from prose_doctor.ml.psychic_distance import analyze_chapter

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    mm = ModelManager()
    for f in files:
        text = f.read_text()
        result = analyze_chapter(text, f.name, mm)

        print(f"\n{'─' * 60}")
        print(f" {result.filename}")
        print(f" {len(result.sentence_scores)} sentences | "
              f"mean distance: {result.mean_distance:.3f} ({result.label}) | "
              f"std: {result.std_distance:.3f}")
        print(f"{'─' * 60}")

        if result.zoom_jumps:
            print(f"\n  Zoom jumps ({len(result.zoom_jumps)}):")
            for j in result.zoom_jumps:
                direction = "→ closer" if j.delta > 0 else "→ farther"
                print(f"    [{j.paragraph_idx}:{j.sentence_idx}] "
                      f"{j.distance_before:.2f} → {j.distance_after:.2f} "
                      f"(Δ{j.delta:+.2f} {direction})")
                print(f"      {j.text}")
        else:
            print("\n  No zoom jumps detected.")

        # Paragraph-level curve (compact)
        if result.paragraph_means:
            print(f"\n  Distance curve (per paragraph):")
            bar_width = 40
            for pi, pm in enumerate(result.paragraph_means):
                filled = int(pm * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)
                print(f"    {pi:>3} [{bar}] {pm:.2f}")


def cmd_contour(args: argparse.Namespace) -> None:
    """Run information contour analysis."""
    try:
        from prose_doctor.ml import require_ml

        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    from prose_doctor.ml.info_contour import analyze_chapter
    from prose_doctor.ml.models import ModelManager

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    mm = ModelManager()
    for f in files:
        text = f.read_text()
        result = analyze_chapter(text, f.name, mm)

        print(f"\n{'─' * 60}")
        print(f" {result.filename}")
        print(f" {result.sentence_count} sentences | "
              f"mean surprisal: {result.mean_surprisal:.3f} | "
              f"CV: {result.cv_surprisal:.3f} | {result.label}")
        print(f" Dominant cycle: ~{result.dominant_period} sentences "
              f"(~{result.dominant_period_words} words) | "
              f"rhythmicity: {result.rhythmicity:.3f} | "
              f"spectral entropy: {result.spectral_entropy:.3f}")
        print(f"{'─' * 60}")

        if result.flatlines:
            print(f"\n  Information flatlines ({len(result.flatlines)}):")
            for fl in result.flatlines:
                print(f"    sentences {fl['start']}-{fl['end']} "
                      f"({fl['length']} sent, mean={fl['mean_surprisal']:.3f})")

        if result.spikes:
            print(f"\n  Surprisal spikes ({len(result.spikes)}):")
            for sp in result.spikes[:5]:
                print(f"    [{sp['index']}] surprisal={sp['surprisal']:.3f} "
                      f"(z={sp['z_score']:.1f})")
                print(f"      {sp['text']}")


def cmd_sensory(args: argparse.Namespace) -> None:
    """Run sensory modality profiler."""
    try:
        from prose_doctor.ml import require_ml

        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    from prose_doctor.ml.models import ModelManager
    from prose_doctor.ml.sensory import profile_chapter, MODALITIES

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    mm = ModelManager()
    for f in files:
        text = f.read_text()
        result = profile_chapter(text, f.name, mm)

        print(f"\n{'─' * 60}")
        print(f" {result.filename} ({result.word_count} words)")
        print(f" Dominant: {result.dominant_modality} | "
              f"Weakest: {result.weakest_modality} | "
              f"Balance: {result.balance_ratio:.2f}")
        print(f"{'─' * 60}")

        # Bar chart
        max_score = max(result.scores.values()) or 1
        for mod, score in result.scores.items():
            bar_width = 30
            filled = int(score / max_score * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"  {mod:<15} [{bar}] {score:.3f}")

        if result.deserts:
            print(f"\n  Sensory deserts ({len(result.deserts)}):")
            for d in result.deserts:
                print(f"    paragraphs {d['start']}-{d['end']} ({d['length']} paragraphs)")

        if result.prescription:
            print(f"\n  Rx: {result.prescription}")


def cmd_critique(args: argparse.Namespace) -> None:
    """Generate a structured revision prompt."""
    try:
        from prose_doctor.ml import require_ml

        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    from prose_doctor.critique import build_critique, format_critique_prompt
    from prose_doctor.ml.models import ModelManager

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    config = ProjectConfig.load(files[0].parent)
    mm = ModelManager()

    for f in files:
        text = f.read_text()
        report = diagnose(text, filename=f.name, config=config)

        # Run ML analyzers (same as scan --deep)
        from prose_doctor.ml.psychic_distance import analyze_chapter as pd_analyze
        pd = pd_analyze(text, f.name, mm)
        report.psychic_distance = {
            "mean_distance": pd.mean_distance,
            "std_distance": pd.std_distance,
            "label": pd.label,
            "zoom_jumps": len(pd.zoom_jumps),
            "paragraph_means": [round(m, 3) for m in pd.paragraph_means],
        }

        from prose_doctor.ml.info_contour import analyze_chapter as ic_analyze
        ic = ic_analyze(text, f.name, mm)
        report.info_contour = {
            "mean_surprisal": ic.mean_surprisal,
            "cv_surprisal": ic.cv_surprisal,
            "label": ic.label,
            "dominant_period": ic.dominant_period,
            "dominant_period_words": ic.dominant_period_words,
            "rhythmicity": ic.rhythmicity,
            "flatlines": ic.flatlines,  # full detail for retexture
            "spikes": len(ic.spikes),
        }

        from prose_doctor.ml.foregrounding import score_chapter
        fg = score_chapter(text, f.name, mm)
        report.foregrounding = {
            "index": round(fg.index, 2),
            "alliteration_per_1k": round(fg.alliteration, 1),
            "inversion_pct": round(fg.inversion_pct, 1),
            "sentence_length_cv": round(fg.sl_cv, 2),
            "fragment_pct": round(fg.fragment_pct, 1),
            "weakest_axis": fg.weakest_axis,
            "prescription": fg.prescription,
        }

        from prose_doctor.ml.sensory import profile_chapter
        sp = profile_chapter(text, f.name, mm)
        report.sensory = {
            "dominant": sp.dominant_modality,
            "weakest": sp.weakest_modality,
            "balance": round(sp.balance_ratio, 3),
            "scores": sp.scores,
            "deserts": len(sp.deserts),
            "prescription": sp.prescription,
        }

        # Run twins across all files for self-referential feedback
        twins_data = None
        if len(files) > 1:
            from prose_doctor.ml.twins import find_twins
            twins = find_twins(files, mm, max_results=5)
            twins_data = [
                {
                    "flat_idx": tw.flat_idx,
                    "flat_text": tw.flat_text,
                    "flat_texture": tw.flat_texture,
                    "twin_idx": tw.twin_idx,
                    "twin_text": tw.twin_text,
                    "twin_texture": tw.twin_texture,
                }
                for tw in twins
                if tw.flat_file == f.name
            ]

        # Build and format critique
        sections = build_critique(report.to_dict(), twins=twins_data)
        prompt = format_critique_prompt(f.name, sections, word_count=report.word_count)
        print(prompt)

        # Retexture pass: generate sensory fragments for flat zones
        if getattr(args, "retexture", False):
            from prose_doctor.retexture import retexture_chapter

            endpoint = getattr(args, "endpoint", None) or "http://localhost:8081/v1"
            model = getattr(args, "model", None) or "bartowski-drummer-24b-q6k_l"

            suggestions = retexture_chapter(
                text, report.to_dict(),
                n_variants=5, max_candidates=5,
                endpoint=endpoint, model=model,
            )

            if suggestions:
                print("\n## Sensory Fragments for Injection\n")
                print("Generated by sensory consultant LLM. Shape to fit POV voice.\n")
                for sg in suggestions:
                    c = sg.candidate
                    print(f"**Paragraph {c.paragraph_idx}** ({c.reason}, score={c.score:.3f}):")
                    print(f"> {c.text[:150]}...\n")
                    print(f"*Best fragment:*\n{sg.best}\n")
                    if len(sg.fragments) > 1:
                        print(f"<details><summary>{len(sg.fragments)} variants</summary>\n")
                        for i, frag in enumerate(sg.fragments):
                            print(f"{i+1}. {frag}\n")
                        print("</details>\n")

        print()


def cmd_classify(args: argparse.Namespace) -> None:
    """Run ML classifier on files."""
    try:
        from prose_doctor.ml import require_ml

        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    from prose_doctor.ml.models import ModelManager
    from prose_doctor.ml.slop_scorer import SlopScorer

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    config = ProjectConfig.load(files[0].parent)
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint:
        config.slop_classifier_model = checkpoint
    mm = ModelManager()
    scorer = SlopScorer(model_manager=mm, config=config)

    threshold = getattr(args, "threshold", 0.5)
    top_n = getattr(args, "top", 20)

    for path in files:
        text = path.read_text()
        stats = scorer.chapter_stats(text, threshold=threshold)

        print(f"\n{'─' * 60}")
        print(f" {path.name}")
        print(
            f" {stats['total_paragraphs']} paragraphs | "
            f"{stats['flagged_count']} flagged ({stats['flagged_pct']:.0f}%) | "
            f"mean={stats['mean_slop']:.3f} | max={stats['max_slop']:.3f}"
        )
        print(f"{'─' * 60}")

        for s in stats["scored"][:top_n]:
            if s["slop_prob"] < threshold:
                break
            short = s["text"][:80].replace("\n", " ")
            class_tag = s.get("class_name", "")
            rules = s.get("rules", [])
            rule_tag = f" (rule:{rules[0]})" if rules else ""
            if class_tag and class_tag != "clean":
                print(
                    f"  [{s['index']:>3}] {s['slop_prob']:.2f} "
                    f"{class_tag:<15}{rule_tag} {short}"
                )
            else:
                print(f"  [{s['index']:>3}] {s['slop_prob']:.2f} {rule_tag} {short}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="prose-doctor",
        description="Prose analysis toolkit: patterns, vocabulary, density, and ML scoring",
    )
    subparsers = parser.add_subparsers(dest="command")

    # scan
    scan_p = subparsers.add_parser("scan", help="Scan files for prose patterns")
    scan_p.add_argument("files", nargs="+", help="Files or directories to scan")
    scan_p.add_argument("--deep", action="store_true", help="Run ML analysis too")
    scan_p.add_argument("--json", action="store_true", help="JSON output")

    # init
    subparsers.add_parser("init", help="Generate .prose-doctor.toml template")

    # index (ML)
    index_p = subparsers.add_parser("index", help="Foregrounding index [ML]")
    index_p.add_argument("files", nargs="+", help="Files or directories")

    # twins (ML)
    twins_p = subparsers.add_parser("twins", help="Find textured twins [ML]")
    twins_p.add_argument("files", nargs="+", help="Files or directories")
    twins_p.add_argument("--top", "-n", type=int, default=20, help="Max results")

    # classify (ML)
    classify_p = subparsers.add_parser("classify", help="ML classifier scores [ML]")
    classify_p.add_argument("files", nargs="+", help="Files or directories")
    classify_p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Model checkpoint path or HuggingFace repo"
    )
    classify_p.add_argument(
        "--threshold", "-t", type=float, default=0.5, help="Score threshold"
    )
    classify_p.add_argument(
        "--top", "-n", type=int, default=20, help="Top N per chapter"
    )

    # distance (ML)
    dist_p = subparsers.add_parser("distance", help="Psychic distance analysis [ML]")
    dist_p.add_argument("files", nargs="+", help="Files or directories")

    # contour (ML)
    contour_p = subparsers.add_parser("contour", help="Information contour analysis [ML]")
    contour_p.add_argument("files", nargs="+", help="Files or directories")

    # sensory (ML)
    sens_p = subparsers.add_parser("sensory", help="Sensory modality profiler [ML]")
    sens_p.add_argument("files", nargs="+", help="Files or directories")

    # critique (ML)
    crit_p = subparsers.add_parser("critique", help="Generate revision prompt [ML]")
    crit_p.add_argument("files", nargs="+", help="Files or directories")
    crit_p.add_argument("--retexture", action="store_true",
                        help="Generate sensory fragments for flat zones via LLM")
    crit_p.add_argument("--endpoint", type=str, default=None,
                        help="LLM endpoint for retexture (default: localhost:8081)")
    crit_p.add_argument("--model", type=str, default=None,
                        help="Model name for retexture (default: bartowski-drummer-24b-q6k_l)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handlers = {
        "scan": cmd_scan,
        "init": cmd_init,
        "index": cmd_index,
        "twins": cmd_twins,
        "classify": cmd_classify,
        "distance": cmd_distance,
        "contour": cmd_contour,
        "sensory": cmd_sensory,
        "critique": cmd_critique,
    }
    handlers[args.command](args)
