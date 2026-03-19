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

                # Classifier (uses HF Hub default or config override)
                try:
                    from prose_doctor.ml.slop_scorer import SlopScorer

                    scorer = SlopScorer(model_manager=mm, config=config)
                    stats = scorer.chapter_stats(text)
                    report.pattern_hits.extend(
                        {
                            "pattern": f"ml:{s['class_name']}",
                            "line": 0,
                            "match": s["text"][:80],
                            "severity": "ml",
                        }
                        for s in stats.get("scored", [])
                        if s["slop_prob"] > 0.5
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
    }
    handlers[args.command](args)
