"""Reusable deep-scan function for the revision agent."""
from __future__ import annotations

from prose_doctor.agent_models import ProseMetrics
from prose_doctor.config import ProjectConfig


def scan_deep(
    text: str,
    filename: str = "chapter.md",
    config: ProjectConfig | None = None,
    metrics_only: bool = False,
    previous_report: dict | None = None,
) -> tuple[ProseMetrics, dict]:
    """Run the ML analyzer suite and return structured metrics + raw report.

    Args:
        text: Chapter text to analyze.
        filename: For reporting.
        config: Project config.
        metrics_only: If True, skip expensive analyzers that don't feed into
            ProseMetrics (sensory profiler, pattern scanner, vocabulary crutches).
            Use this for subsequent scans during the revision loop.
        previous_report: If provided, carry forward non-metric fields from
            the previous scan (sensory, pattern_hits, etc) so the report
            stays complete for critique.

    Returns (ProseMetrics, report_dict).
    """
    from prose_doctor.ml import require_ml
    require_ml()
    from prose_doctor.ml.models import ModelManager

    cfg = config or ProjectConfig()
    mm = ModelManager()

    if metrics_only:
        # Lightweight: skip diagnose() and sensory — just compute the 8 metrics
        from prose_doctor.analyzers.doctor import ChapterHealth
        from prose_doctor.text import count_words
        report = ChapterHealth(
            filename=filename,
            word_count=count_words(text),
        )
    else:
        from prose_doctor.analyzers.doctor import diagnose
        report = diagnose(text, filename=filename, config=cfg)

    # Psychic distance — always run (feeds pd_mean, pd_std)
    from prose_doctor.ml.psychic_distance import analyze_chapter as pd_analyze
    pd = pd_analyze(text, filename, mm)
    report.psychic_distance = {
        "mean_distance": pd.mean_distance,
        "std_distance": pd.std_distance,
        "label": pd.label,
        "zoom_jumps": len(pd.zoom_jumps),
        "paragraph_means": [round(m, 3) for m in pd.paragraph_means],
    }

    # Information contour — always run (feeds ic_rhythmicity, ic_spikes, ic_flatlines)
    from prose_doctor.ml.info_contour import analyze_chapter as ic_analyze
    ic = ic_analyze(text, filename, mm)
    report.info_contour = {
        "mean_surprisal": ic.mean_surprisal,
        "cv_surprisal": ic.cv_surprisal,
        "label": ic.label,
        "dominant_period": ic.dominant_period,
        "dominant_period_words": ic.dominant_period_words,
        "rhythmicity": ic.rhythmicity,
        "flatlines": ic.flatlines,
        "spikes": len(ic.spikes),
    }

    # Foregrounding — always run (feeds fg_inversion, fg_sl_cv, fg_fragment)
    from prose_doctor.ml.foregrounding import score_chapter
    fg = score_chapter(text, filename, mm)
    report.foregrounding = {
        "index": round(fg.index, 2),
        "inversion_pct": round(fg.inversion_pct, 1),
        "sentence_length_cv": round(fg.sl_cv, 2),
        "fragment_pct": round(fg.fragment_pct, 1),
        "weakest_axis": fg.weakest_axis,
        "prescription": fg.prescription,
    }

    if not metrics_only:
        # Sensory profiler — only on full scans
        from prose_doctor.ml.sensory import profile_chapter
        sp = profile_chapter(text, filename, mm)
        report.sensory = {
            "dominant": sp.dominant_modality,
            "weakest": sp.weakest_modality,
            "balance": round(sp.balance_ratio, 3),
            "scores": sp.scores,
            "deserts": len(sp.deserts),
            "prescription": sp.prescription,
        }
    elif previous_report:
        # Carry forward sensory from previous full scan
        report.sensory = previous_report.get("sensory")

    report_dict = report.to_dict()

    # Carry forward non-metric fields from previous report
    if metrics_only and previous_report:
        for key in ("dialogue", "pacing", "vocabulary_crutches", "pattern_hits"):
            if key in previous_report and key not in report_dict:
                report_dict[key] = previous_report[key]

    # Build ProseMetrics from report
    ic_dict = report_dict.get("info_contour") or {}
    spikes_val = ic_dict.get("spikes", 0)
    if isinstance(spikes_val, list):
        spikes_val = len(spikes_val)
    flatlines_val = ic_dict.get("flatlines", 0)
    if isinstance(flatlines_val, list):
        flatlines_val = len(flatlines_val)

    fg_dict = report_dict.get("foregrounding") or {}
    pd_dict = report_dict.get("psychic_distance") or {}

    metrics = ProseMetrics(
        pd_mean=pd_dict.get("mean_distance", 0),
        pd_std=pd_dict.get("std_distance", 0),
        fg_inversion=fg_dict.get("inversion_pct", 0),
        fg_sl_cv=fg_dict.get("sentence_length_cv", 0),
        fg_fragment=fg_dict.get("fragment_pct", 0),
        ic_rhythmicity=ic_dict.get("rhythmicity", 0),
        ic_spikes=spikes_val,
        ic_flatlines=flatlines_val,
    )

    return metrics, report_dict
