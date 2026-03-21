"""CLI dispatcher for prose-doctor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prose_doctor.analyzers.doctor import ChapterHealth, diagnose
from prose_doctor.config import ProjectConfig
from prose_doctor.output.json_output import reports_to_json
from prose_doctor.output.table import format_chapter_report, format_summary


def _resolve_tier_filter(args: argparse.Namespace) -> str:
    """Map --experimental / --validated flags to a tier filter string.

    Default (no flag): 'stable' — run only stable lenses.
    --validated: run validated + stable lenses.
    --experimental: run all lenses including experimental.
    """
    if getattr(args, "experimental", False):
        return "experimental"
    if getattr(args, "validated", False):
        return "validated"
    return "stable"


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
                from prose_doctor.providers import ProviderPool, require_ml

                require_ml()
                providers = ProviderPool()

                from prose_doctor.lenses.defaults import default_registry
                from prose_doctor.lenses.runner import LensRunner
                from prose_doctor.validation.tiers import load_tiers

                registry = default_registry()
                tier_filter = _resolve_tier_filter(args)
                runner = LensRunner(registry, providers, tier_filter=tier_filter, tiers=load_tiers())

                # Classifier with density budgets — only report classes
                # that exceed a per-1000-word threshold. Individual hits
                # are candidates; accumulation makes them findings.
                try:
                    from prose_doctor.lenses.slop_classifier import SlopScorer

                    scorer = SlopScorer(config=config)
                    scored = scorer.score_paragraphs(text)
                    word_count = report.word_count or 1

                    # Count hits per class
                    class_hits: dict[str, list[dict]] = {}
                    for s in scored:
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

                # Foregrounding
                fg = runner.run_one("foregrounding", text, f.name)
                fg_ch = fg.per_chapter or {}
                report.foregrounding = {
                    "index": round(fg_ch.get("index", 0), 2),
                    "alliteration_per_1k": round(fg_ch.get("alliteration", 0), 1),
                    "inversion_pct": round(fg_ch.get("inversion_pct", 0), 1),
                    "sentence_length_cv": round(fg_ch.get("sl_cv", 0), 2),
                    "fragment_pct": round(fg_ch.get("fragment_pct", 0), 1),
                    "weakest_axis": fg.raw.get("weakest_axis", ""),
                    "prescription": fg.raw.get("prescription", ""),
                }

                # Perplexity
                ppl = runner.run_one("perplexity", text, f.name)
                ppl_ch = ppl.per_chapter or {}
                ppl_raw = ppl.raw or {}
                report.perplexity = {
                    **ppl_raw.get("stats", {}),
                    "smoothest_paragraphs": ppl_raw.get("smoothest_paragraphs", []),
                }

                # Emotion arc
                emo = runner.run_one("emotion_arc", text, f.name)
                emo_ch = emo.per_chapter or {}
                emo_raw = emo.raw or {}
                report.emotion = {
                    "flat": bool(emo_ch.get("flat", 0)),
                    "arc": emo_raw.get("arc", "?"),
                    "std": emo_raw.get("stats", {}).get("std", 0),
                    "dynamic_range": emo_ch.get("dynamic_range", 0),
                }

                # Psychic distance
                pd_r = runner.run_one("psychic_distance", text, f.name)
                pd_ch = pd_r.per_chapter or {}
                pd_raw = pd_r.raw or {}
                pd_para = pd_r.per_paragraph or {}
                report.psychic_distance = {
                    "mean_distance": pd_ch.get("pd_mean", 0),
                    "std_distance": pd_ch.get("pd_std", 0),
                    "label": pd_raw.get("label", ""),
                    "zoom_jumps": len(pd_raw.get("zoom_jumps", [])),
                    "paragraph_means": pd_para.get("pd_mean", []),
                }

                # Information contour
                ic = runner.run_one("info_contour", text, f.name)
                ic_ch = ic.per_chapter or {}
                ic_raw = ic.raw or {}
                report.info_contour = {
                    "mean_surprisal": ic_ch.get("mean_surprisal", 0),
                    "cv_surprisal": ic_ch.get("cv_surprisal", 0),
                    "label": ic_raw.get("label", ""),
                    "dominant_period": ic_ch.get("dominant_period", 0),
                    "dominant_period_words": ic_raw.get("dominant_period_words", 0),
                    "rhythmicity": ic_ch.get("rhythmicity", 0),
                    "flatlines": int(ic_ch.get("flatlines", 0)),
                    "spikes": int(ic_ch.get("spikes", 0)),
                }

                # Sensory profiler
                sp = runner.run_one("sensory", text, f.name)
                sp_ch = sp.per_chapter or {}
                sp_raw = sp.raw or {}
                report.sensory = {
                    "dominant": sp_ch.get("dominant_modality", ""),
                    "weakest": sp_ch.get("weakest_modality", ""),
                    "balance": round(sp_ch.get("balance_ratio", 0), 3),
                    "scores": {
                        m: sp_ch.get(m, 0)
                        for m in ["visual", "auditory", "haptic",
                                  "olfactory", "gustatory", "interoceptive"]
                    },
                    "deserts": len(sp_raw.get("deserts", [])),
                    "prescription": sp_raw.get("prescription", ""),
                }

                # Dialogue voice separation
                dl = runner.run_one("dialogue_voice", text, f.name)
                dl_ch = dl.per_chapter or {}
                dl_raw = dl.raw or {}
                report.dialogue = {
                    "dialogue_ratio": dl_ch.get("dialogue_ratio", 0),
                    "speakers": dl_raw.get("speakers", {}),
                    "speaker_separation": dl_ch.get("speaker_separation", 0),
                    "speaker_similarities": dl_raw.get("speaker_similarities", {}),
                    "longest_dialogue_run": dl_raw.get("longest_dialogue_run", 0),
                    "longest_narration_run": dl_raw.get("longest_narration_run", 0),
                    "all_same_voice": dl_raw.get("all_same_voice", False),
                    "talking_heads": dl_raw.get("longest_dialogue_run", 0),
                    "prescription": dl_raw.get("prescription", ""),
                }

                # Scene pacing
                pc = runner.run_one("pacing", text, f.name)
                pc_ch = pc.per_chapter or {}
                pc_raw = pc.raw or {}
                mode_ratios = {
                    k.replace("_ratio", ""): v
                    for k, v in pc_ch.items()
                    if k.endswith("_ratio")
                }
                report.pacing = {
                    "mode_ratios": mode_ratios,
                    "longest_runs": pc_raw.get("longest_runs", {}),
                    "dominant_mode": max(mode_ratios, key=mode_ratios.get) if mode_ratios else "unknown",
                    "talking_heads": len(pc_raw.get("talking_heads", [])),
                    "action_deserts": len(pc_raw.get("action_deserts", [])),
                    "interiority_gaps": len(pc_raw.get("interiority_gaps", [])),
                    "prescription": "",  # computed inline below
                }
                # Build prescription for pacing
                _pacing_issues = []
                if pc_raw.get("talking_heads"):
                    _pacing_issues.append(
                        f"{len(pc_raw['talking_heads'])} talking-head runs — "
                        f"break up with action or interiority."
                    )
                if pc_raw.get("action_deserts"):
                    _pacing_issues.append(
                        f"{len(pc_raw['action_deserts'])} action deserts — "
                        f"add physical grounding."
                    )
                if pc_raw.get("interiority_gaps"):
                    _pacing_issues.append(
                        f"{len(pc_raw['interiority_gaps'])} interiority gaps — "
                        f"add thought/feeling beats."
                    )
                report.pacing["prescription"] = " ".join(_pacing_issues)

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
        from prose_doctor.providers import require_ml

        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    from prose_doctor.lenses.defaults import default_registry
    from prose_doctor.lenses.runner import LensRunner
    from prose_doctor.providers import ProviderPool

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    providers = ProviderPool()
    registry = default_registry()
    runner = LensRunner(registry, providers)

    results = []
    for f in files:
        text = f.read_text()
        r = runner.run_one("foregrounding", text, f.name)
        ch = r.per_chapter or {}
        results.append((f.name, ch, r.raw))

    results.sort(key=lambda x: x[1].get("index", 0))

    header = (
        f"{'Chapter':<42} {'Words':>5} {'Allit':>5} {'Inv%':>5} "
        f"{'SLCV':>5} {'Frg%':>5} {'INDEX':>6}  Weakest"
    )
    print(header)
    print("-" * len(header))
    for filename, ch, raw in results:
        print(
            f"{filename:<42} {int(ch.get('word_count', 0)):>5} {ch.get('alliteration', 0):>5.1f} "
            f"{ch.get('inversion_pct', 0):>4.1f}% {ch.get('sl_cv', 0):>5.2f} {ch.get('fragment_pct', 0):>4.1f}% "
            f"{ch.get('index', 0):>6.2f}  {raw.get('weakest_axis', '')}"
        )


def cmd_twins(args: argparse.Namespace) -> None:
    """Run twin-finder analysis."""
    try:
        from prose_doctor.providers import require_ml

        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    from prose_doctor.lenses.twins import find_twins
    from prose_doctor.providers import ProviderPool

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    providers = ProviderPool()
    twins = find_twins(files, providers, max_results=getattr(args, "top", 20))

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
        from prose_doctor.providers import require_ml

        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    from prose_doctor.lenses.defaults import default_registry
    from prose_doctor.lenses.runner import LensRunner
    from prose_doctor.providers import ProviderPool

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    providers = ProviderPool()
    registry = default_registry()
    runner = LensRunner(registry, providers)

    for f in files:
        text = f.read_text()
        lr = runner.run_one("psychic_distance", text, f.name)
        result = lr.raw.get("result")  # PsychicDistanceResult object

        print(f"\n{'─' * 60}")
        print(f" {f.name}")
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
        from prose_doctor.providers import require_ml

        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    from prose_doctor.lenses.defaults import default_registry
    from prose_doctor.lenses.runner import LensRunner
    from prose_doctor.providers import ProviderPool

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    providers = ProviderPool()
    registry = default_registry()
    runner = LensRunner(registry, providers)

    for f in files:
        text = f.read_text()
        lr = runner.run_one("info_contour", text, f.name)
        ch = lr.per_chapter or {}
        raw = lr.raw or {}

        print(f"\n{'─' * 60}")
        print(f" {f.name}")
        print(f" {raw.get('sentence_count', 0)} sentences | "
              f"mean surprisal: {ch.get('mean_surprisal', 0):.3f} | "
              f"CV: {ch.get('cv_surprisal', 0):.3f} | {raw.get('label', '')}")
        print(f" Dominant cycle: ~{int(ch.get('dominant_period', 0))} sentences "
              f"(~{raw.get('dominant_period_words', 0)} words) | "
              f"rhythmicity: {ch.get('rhythmicity', 0):.3f} | "
              f"spectral entropy: {ch.get('spectral_entropy', 0):.3f}")
        print(f"{'─' * 60}")

        flatlines = raw.get("flatline_details", [])
        if flatlines:
            print(f"\n  Information flatlines ({len(flatlines)}):")
            for fl in flatlines:
                print(f"    sentences {fl['start']}-{fl['end']} "
                      f"({fl['length']} sent, mean={fl['mean_surprisal']:.3f})")

        spikes = raw.get("spike_details", [])
        if spikes:
            print(f"\n  Surprisal spikes ({len(spikes)}):")
            for sp in spikes[:5]:
                print(f"    [{sp['index']}] surprisal={sp['surprisal']:.3f} "
                      f"(z={sp['z_score']:.1f})")
                print(f"      {sp['text']}")


def cmd_sensory(args: argparse.Namespace) -> None:
    """Run sensory modality profiler."""
    try:
        from prose_doctor.providers import require_ml

        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    from prose_doctor.lenses.defaults import default_registry
    from prose_doctor.lenses.runner import LensRunner
    from prose_doctor.lenses.sensory import MODALITIES
    from prose_doctor.providers import ProviderPool

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    providers = ProviderPool()
    registry = default_registry()
    runner = LensRunner(registry, providers)

    for f in files:
        text = f.read_text()
        lr = runner.run_one("sensory", text, f.name)
        ch = lr.per_chapter or {}
        raw = lr.raw or {}

        print(f"\n{'─' * 60}")
        print(f" {f.name} ({raw.get('word_count', 0)} words)")
        print(f" Dominant: {ch.get('dominant_modality', '')} | "
              f"Weakest: {ch.get('weakest_modality', '')} | "
              f"Balance: {ch.get('balance_ratio', 0):.2f}")
        print(f"{'─' * 60}")

        # Bar chart
        scores = {m: ch.get(m, 0) for m in MODALITIES}
        max_score = max(scores.values()) or 1
        for mod, score in scores.items():
            bar_width = 30
            filled = int(score / max_score * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"  {mod:<15} [{bar}] {score:.3f}")

        deserts = raw.get("deserts", [])
        if deserts:
            print(f"\n  Sensory deserts ({len(deserts)}):")
            for d in deserts:
                print(f"    paragraphs {d['start']}-{d['end']} ({d['length']} paragraphs)")

        prescription = raw.get("prescription", "")
        if prescription:
            print(f"\n  Rx: {prescription}")


def cmd_critique(args: argparse.Namespace) -> None:
    """Generate a structured revision prompt."""
    try:
        from prose_doctor.providers import require_ml

        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    from prose_doctor.critique import build_critique, format_critique_prompt
    from prose_doctor.lenses.defaults import default_registry
    from prose_doctor.lenses.runner import LensRunner
    from prose_doctor.providers import ProviderPool

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    config = ProjectConfig.load(files[0].parent)
    providers = ProviderPool()
    registry = default_registry()
    runner = LensRunner(registry, providers)

    for f in files:
        text = f.read_text()
        report = diagnose(text, filename=f.name, config=config)

        # Run ML analyzers via lens framework
        pd_r = runner.run_one("psychic_distance", text, f.name)
        pd_ch = pd_r.per_chapter or {}
        pd_raw = pd_r.raw or {}
        pd_para = pd_r.per_paragraph or {}
        report.psychic_distance = {
            "mean_distance": pd_ch.get("pd_mean", 0),
            "std_distance": pd_ch.get("pd_std", 0),
            "label": pd_raw.get("label", ""),
            "zoom_jumps": len(pd_raw.get("zoom_jumps", [])),
            "paragraph_means": pd_para.get("pd_mean", []),
        }

        ic = runner.run_one("info_contour", text, f.name)
        ic_ch = ic.per_chapter or {}
        ic_raw = ic.raw or {}
        report.info_contour = {
            "mean_surprisal": ic_ch.get("mean_surprisal", 0),
            "cv_surprisal": ic_ch.get("cv_surprisal", 0),
            "label": ic_raw.get("label", ""),
            "dominant_period": ic_ch.get("dominant_period", 0),
            "dominant_period_words": ic_raw.get("dominant_period_words", 0),
            "rhythmicity": ic_ch.get("rhythmicity", 0),
            "flatlines": ic_raw.get("flatline_details", []),  # full detail for retexture
            "spikes": int(ic_ch.get("spikes", 0)),
        }

        fg = runner.run_one("foregrounding", text, f.name)
        fg_ch = fg.per_chapter or {}
        report.foregrounding = {
            "index": round(fg_ch.get("index", 0), 2),
            "alliteration_per_1k": round(fg_ch.get("alliteration", 0), 1),
            "inversion_pct": round(fg_ch.get("inversion_pct", 0), 1),
            "sentence_length_cv": round(fg_ch.get("sl_cv", 0), 2),
            "fragment_pct": round(fg_ch.get("fragment_pct", 0), 1),
            "weakest_axis": fg.raw.get("weakest_axis", ""),
            "prescription": fg.raw.get("prescription", ""),
        }

        sp = runner.run_one("sensory", text, f.name)
        sp_ch = sp.per_chapter or {}
        sp_raw = sp.raw or {}
        report.sensory = {
            "dominant": sp_ch.get("dominant_modality", ""),
            "weakest": sp_ch.get("weakest_modality", ""),
            "balance": round(sp_ch.get("balance_ratio", 0), 3),
            "scores": {
                m: sp_ch.get(m, 0)
                for m in ["visual", "auditory", "haptic",
                          "olfactory", "gustatory", "interoceptive"]
            },
            "deserts": len(sp_raw.get("deserts", [])),
            "prescription": sp_raw.get("prescription", ""),
        }

        # Dialogue voice separation
        dl = runner.run_one("dialogue_voice", text, f.name)
        dl_ch = dl.per_chapter or {}
        dl_raw = dl.raw or {}
        report.dialogue = {
            "dialogue_ratio": dl_ch.get("dialogue_ratio", 0),
            "speakers": dl_raw.get("speakers", {}),
            "speaker_separation": dl_ch.get("speaker_separation", 0),
            "all_same_voice": dl_raw.get("all_same_voice", False),
            "talking_heads": dl_raw.get("longest_dialogue_run", 0),
            "prescription": dl_raw.get("prescription", ""),
        }

        # Scene pacing
        pc = runner.run_one("pacing", text, f.name)
        pc_ch = pc.per_chapter or {}
        pc_raw = pc.raw or {}
        mode_ratios = {
            k.replace("_ratio", ""): v
            for k, v in pc_ch.items()
            if k.endswith("_ratio")
        }
        _pacing_issues = []
        if pc_raw.get("talking_heads"):
            _pacing_issues.append(
                f"{len(pc_raw['talking_heads'])} talking-head runs."
            )
        if pc_raw.get("action_deserts"):
            _pacing_issues.append(
                f"{len(pc_raw['action_deserts'])} action deserts."
            )
        if pc_raw.get("interiority_gaps"):
            _pacing_issues.append(
                f"{len(pc_raw['interiority_gaps'])} interiority gaps."
            )
        report.pacing = {
            "mode_ratios": mode_ratios,
            "talking_heads": len(pc_raw.get("talking_heads", [])),
            "action_deserts": len(pc_raw.get("action_deserts", [])),
            "interiority_gaps": len(pc_raw.get("interiority_gaps", [])),
            "prescription": " ".join(_pacing_issues),
        }

        # Run twins across all files for self-referential feedback
        twins_data = None
        if len(files) > 1:
            from prose_doctor.lenses.twins import find_twins

            twins = find_twins(files, providers, max_results=5)
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
                print("\n## Texture Fragments for Injection\n")
                print("Generated by creative LLM. Shape to fit POV voice.\n")
                for sg in suggestions:
                    c = sg.candidate
                    mode_labels = {
                        "sensory": "body report",
                        "collocation": "defamiliarized",
                        "interiority": "interior monologue",
                        "rhythm": "restructured",
                    }
                    mode_label = mode_labels.get(c.mode, c.mode)
                    print(f"**Paragraph {c.paragraph_idx}** [{mode_label}] ({c.reason}, score={c.score:.3f}):")
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
        from prose_doctor.providers import require_ml

        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    from prose_doctor.lenses.slop_classifier import SlopScorer

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    config = ProjectConfig.load(files[0].parent)
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint:
        config.slop_classifier_model = checkpoint
    scorer = SlopScorer(config=config)

    threshold = getattr(args, "threshold", 0.5)
    top_n = getattr(args, "top", 20)

    for path in files:
        text = path.read_text()
        scored = scorer.score_paragraphs(text)

        # Compute summary stats (replaces chapter_stats)
        total = len(scored)
        flagged = [s for s in scored if s["slop_prob"] > threshold]
        flagged_pct = len(flagged) / total * 100 if total else 0.0
        mean_slop = sum(s["slop_prob"] for s in scored) / total if total else 0.0
        max_slop = scored[0]["slop_prob"] if scored else 0.0

        print(f"\n{'─' * 60}")
        print(f" {path.name}")
        print(
            f" {total} paragraphs | "
            f"{len(flagged)} flagged ({flagged_pct:.0f}%) | "
            f"mean={mean_slop:.3f} | max={max_slop:.3f}"
        )
        print(f"{'─' * 60}")

        for s in scored[:top_n]:
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


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate lenses against human/LLM corpus."""
    from prose_doctor.providers import require_ml, ProviderPool
    from prose_doctor.lenses.defaults import default_registry
    from prose_doctor.lenses.runner import LensRunner
    from prose_doctor.validation.corpus import load_corpus
    from prose_doctor.validation.discriminator import compute_discrimination
    from prose_doctor.validation.promotion import check_tier
    from prose_doctor.validation.tiers import load_tiers, write_tiers

    require_ml()
    pool = ProviderPool()
    registry = default_registry()

    # Determine corpus paths
    human_dir = Path(args.human_corpus) if args.human_corpus else Path("corpus/human")
    llm_dir = Path(args.llm_corpus) if args.llm_corpus else Path("corpus")  # uses subdirs

    # Load corpus
    if not human_dir.exists():
        print(f"Human corpus not found at {human_dir}", file=sys.stderr)
        sys.exit(1)

    human_files = load_corpus(human_dir)
    # LLM corpus: load from all subdirectories under corpus/ that aren't "human"
    llm_files = []
    if llm_dir.exists():
        for subdir in sorted(llm_dir.iterdir()):
            if subdir.is_dir() and subdir.name != "human":
                llm_files.extend(load_corpus(subdir))

    if not human_files or not llm_files:
        print("Need both human and LLM corpus files for validation.", file=sys.stderr)
        sys.exit(1)

    # Determine which lenses to validate
    if args.all_lenses:
        lens_names = registry.all_names()
    elif args.lens:
        lens_names = [args.lens]
    else:
        print("Specify a lens name or --all", file=sys.stderr)
        sys.exit(1)

    runner = LensRunner(registry, pool)
    results_table = []

    for lens_name in lens_names:
        lens = registry.get(lens_name)
        if lens is None:
            print(f"Unknown lens: {lens_name}", file=sys.stderr)
            continue

        print(f"Validating {lens_name}...", file=sys.stderr)

        # Run lens on both corpora, collect per_chapter metrics
        human_scores: dict[str, list[float]] = {}
        llm_scores: dict[str, list[float]] = {}

        for filename, text in human_files:
            try:
                result = runner.run_one(lens_name, text, filename)
                if result.per_chapter:
                    for key, val in result.per_chapter.items():
                        if isinstance(val, (int, float)):
                            human_scores.setdefault(key, []).append(float(val))
            except Exception as e:
                print(f"  Warning: {filename}: {e}", file=sys.stderr)

        for filename, text in llm_files:
            try:
                result = runner.run_one(lens_name, text, filename)
                if result.per_chapter:
                    for key, val in result.per_chapter.items():
                        if isinstance(val, (int, float)):
                            llm_scores.setdefault(key, []).append(float(val))
            except Exception as e:
                print(f"  Warning: {filename}: {e}", file=sys.stderr)

        # Compute discrimination per metric
        best_d = 0.0
        best_p = 1.0
        best_metric = ""
        for metric in human_scores:
            if metric in llm_scores and len(human_scores[metric]) >= 3 and len(llm_scores[metric]) >= 3:
                stats = compute_discrimination(human_scores[metric], llm_scores[metric])
                if abs(stats["cohens_d"]) > abs(best_d):
                    best_d = stats["cohens_d"]
                    best_p = stats["p_value"]
                    best_metric = metric

        tier = check_tier({"cohens_d": best_d, "p_value": best_p}, revision_evidence=[])
        results_table.append((lens_name, best_metric, best_d, best_p, tier))

    # Print table
    print(f"\n{'Lens':<25} {'Best Metric':<20} {'Cohen d':>8} {'p-value':>10} {'Tier':<12}")
    print("-" * 77)
    for name, metric, d, p, tier in results_table:
        print(f"{name:<25} {metric:<20} {d:>8.3f} {p:>10.6f} {tier:<12}")

    # --promote: merge computed tiers into tiers.toml (only upgrade, never downgrade)
    if args.promote and results_table:
        existing = load_tiers()
        tier_rank = {"experimental": 0, "validated": 1, "stable": 2}
        updated = dict(existing)
        changes = []
        for name, _, _, _, new_tier in results_table:
            old_tier = existing.get(name, "experimental")
            if tier_rank.get(new_tier, 0) > tier_rank.get(old_tier, 0):
                updated[name] = new_tier
                changes.append(f"  {name}: {old_tier} -> {new_tier}")
            elif name not in updated:
                updated[name] = new_tier
        if changes:
            write_tiers(updated)
            print(f"\nPromoted {len(changes)} lens(es):")
            for c in changes:
                print(c)
            print("Written to tiers.toml")
        else:
            print("\nNo promotions — all lenses already at or above computed tier.")


def cmd_revise(args: argparse.Namespace) -> None:
    """Run the agentic revision loop."""
    try:
        from prose_doctor.providers import require_ml
        require_ml()
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    files = _discover_files(args.files)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        from prose_doctor.agent_scan import scan_deep
        from prose_doctor.critique import build_critique, format_critique_prompt
        for f in files:
            text = f.read_text()
            metrics, report = scan_deep(text, filename=f.name)
            sections = build_critique(report)
            print(format_critique_prompt(f.name, sections, word_count=report.get("word_count", 0)))
            print(f"\nMetrics: {metrics.distances()}")
            print(f"Total distance: {metrics.total_distance}")
            print(f"Worst: {metrics.worst_metric}")
        return

    from prose_doctor.agent import run_revision

    for f in files:
        text = f.read_text()
        result = run_revision(
            text,
            filename=f.name,
            max_turns=args.max_turns,
            endpoint=args.endpoint,
            model_name=args.model,
            verbose=args.verbose,
        )

        if args.output:
            Path(args.output).write_text(result.final_text)
            print(f"Wrote revised chapter to {args.output}", file=sys.stderr)
        else:
            print(result.final_text)

        print(f"\n--- Revision Summary ---", file=sys.stderr)
        print(f"Turns: {result.turns_used}", file=sys.stderr)
        print(f"Edits accepted: {result.edits_accepted}", file=sys.stderr)
        print(f"Edits rejected: {result.edits_rejected}", file=sys.stderr)
        print(f"Improved: {', '.join(result.metrics_improved) or 'none'}", file=sys.stderr)
        print(f"Worsened: {', '.join(result.metrics_worsened) or 'none'}", file=sys.stderr)
        print(f"Distance: {result.metrics_initial.total_distance:.4f} → {result.metrics_final.total_distance:.4f}", file=sys.stderr)


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
    scan_p.add_argument("--experimental", action="store_true", help="Include experimental lenses")
    scan_p.add_argument("--validated", action="store_true", help="Include validated + stable lenses")

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

    # revise (agent)
    revise_p = subparsers.add_parser("revise", help="Agentic iterative revision [ML+Agent]")
    revise_p.add_argument("files", nargs="+", help="Files to revise")
    revise_p.add_argument("-o", "--output", type=str, default=None,
                          help="Output file (default: stdout)")
    revise_p.add_argument("--max-turns", type=int, default=8,
                          help="Maximum revision turns (default: 8)")
    revise_p.add_argument("--dry-run", action="store_true",
                          help="Scan and critique only, don't rewrite")
    revise_p.add_argument("--verbose", action="store_true",
                          help="Print each turn's metrics delta")
    revise_p.add_argument("--endpoint", type=str, default="http://localhost:8081/v1",
                          help="LLM endpoint")
    revise_p.add_argument("--model", type=str, default="gpt-oss-120b",
                          help="Model name")
    revise_p.add_argument("--experimental", action="store_true", help="Include experimental lenses")
    revise_p.add_argument("--validated", action="store_true", help="Include validated + stable lenses")

    # validate
    validate_p = subparsers.add_parser("validate", help="Validate lenses against human/LLM corpus")
    validate_p.add_argument("lens", nargs="?", default=None, help="Lens to validate (or --all)")
    validate_p.add_argument("--all", action="store_true", dest="all_lenses", help="Validate all lenses")
    validate_p.add_argument("--promote", action="store_true", help="Auto-promote and write tiers.toml")
    validate_p.add_argument("--human-corpus", type=str, default=None, help="Path to human prose corpus")
    validate_p.add_argument("--llm-corpus", type=str, default=None, help="Path to LLM prose corpus")

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
        "revise": cmd_revise,
        "validate": cmd_validate,
    }
    handlers[args.command](args)
