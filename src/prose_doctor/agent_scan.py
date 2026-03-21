"""Reusable deep-scan function for the revision agent."""
from __future__ import annotations

from prose_doctor.agent_models import ProseMetrics
from prose_doctor.config import ProjectConfig


# Lens names that feed into ProseMetrics — always run.
_METRIC_LENSES = {
    "psychic_distance", "info_contour", "foregrounding",
    "discourse_relations", "concreteness", "situation_shifts",
}

# Additional lenses for full scans.
_FULL_LENSES = _METRIC_LENSES | {"sensory", "referential_cohesion"}


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
    from prose_doctor.providers import require_ml, ProviderPool
    from prose_doctor.lenses import LensRegistry
    from prose_doctor.lenses.defaults import default_registry
    from prose_doctor.lenses.runner import LensRunner

    require_ml()

    cfg = config or ProjectConfig()

    # Build a registry scoped to only the lenses we need.
    wanted = _METRIC_LENSES if metrics_only else _FULL_LENSES
    full_registry = default_registry()
    registry = LensRegistry()
    for name in wanted:
        lens = full_registry.get(name)
        if lens is not None:
            registry.register(lens)

    pool = ProviderPool()
    runner = LensRunner(registry, pool)
    results = runner.run_all(text, filename)

    # --- Build report_dict (rule-based health + lens results) ---

    if metrics_only:
        from prose_doctor.analyzers.doctor import ChapterHealth
        from prose_doctor.text import count_words
        report = ChapterHealth(
            filename=filename,
            word_count=count_words(text),
        )
    else:
        from prose_doctor.analyzers.doctor import diagnose
        report = diagnose(text, filename=filename, config=cfg)

    # Attach lens results onto the report object so to_dict() includes them.
    pd_res = results.get("psychic_distance")
    if pd_res:
        pc = pd_res.per_chapter or {}
        report.psychic_distance = {
            "mean_distance": pc.get("pd_mean", 0),
            "std_distance": pc.get("pd_std", 0),
            "label": (pd_res.raw or {}).get("label", ""),
            "zoom_jumps": int(pc.get("zoom_jump_count", 0)),
            "paragraph_means": (pd_res.per_paragraph or {}).get("pd_mean", []),
        }

    ic_res = results.get("info_contour")
    if ic_res:
        pc = ic_res.per_chapter or {}
        report.info_contour = {
            "mean_surprisal": pc.get("mean_surprisal", 0),
            "cv_surprisal": pc.get("cv_surprisal", 0),
            "label": (ic_res.raw or {}).get("label", ""),
            "dominant_period": pc.get("dominant_period", 0),
            "dominant_period_words": (ic_res.raw or {}).get("dominant_period_words", 0),
            "rhythmicity": pc.get("rhythmicity", 0),
            "flatlines": int(pc.get("flatlines", 0)),
            "spikes": int(pc.get("spikes", 0)),
        }

    fg_res = results.get("foregrounding")
    if fg_res:
        pc = fg_res.per_chapter or {}
        report.foregrounding = {
            "index": round(pc.get("index", 0), 2),
            "inversion_pct": round(pc.get("inversion_pct", 0), 1),
            "sentence_length_cv": round(pc.get("sl_cv", 0), 2),
            "fragment_pct": round(pc.get("fragment_pct", 0), 1),
            "weakest_axis": (fg_res.raw or {}).get("weakest_axis", ""),
            "prescription": (fg_res.raw or {}).get("prescription", ""),
        }

    sensory_res = results.get("sensory")
    if sensory_res:
        pc = sensory_res.per_chapter or {}
        raw = sensory_res.raw or {}
        report.sensory = {
            "dominant": pc.get("dominant_modality", ""),
            "weakest": pc.get("weakest_modality", ""),
            "balance": round(pc.get("balance_ratio", 0), 3),
            "scores": {
                m: pc.get(m, 0)
                for m in ["visual", "auditory", "haptic",
                           "olfactory", "gustatory", "interoceptive"]
            },
            "deserts": len(raw.get("deserts", [])),
            "prescription": raw.get("prescription", ""),
        }
    elif previous_report:
        report.sensory = previous_report.get("sensory")

    dr_res = results.get("discourse_relations")
    if dr_res:
        pc = dr_res.per_chapter or {}
        report.discourse_relations = {
            "relation_entropy": pc.get("relation_entropy", 0),
            "implicit_ratio": pc.get("implicit_ratio", 1.0),
            "causal_ratio": pc.get("causal_ratio", 0),
            "contrastive_ratio": pc.get("contrastive_ratio", 0),
            "additive_ratio": pc.get("additive_ratio", 0),
            "temporal_ratio": pc.get("temporal_ratio", 0),
            "additive_only_zones": int(pc.get("additive_only_zones", 0)),
        }

    cn_res = results.get("concreteness")
    if cn_res:
        pc = cn_res.per_chapter or {}
        report.concreteness = {
            "concreteness_mean": pc.get("concreteness_mean", 3.0),
            "abstractness_ratio": pc.get("abstractness_ratio", 0),
            "vague_noun_density": pc.get("vague_noun_density", 0),
        }

    ss_res = results.get("situation_shifts")
    if ss_res:
        pc = ss_res.per_chapter or {}
        report.situation_shifts = {
            "total_shifts": int(pc.get("total_shifts", 0)),
            "time_shifts": int(pc.get("time_shifts", 0)),
            "space_shifts": int(pc.get("space_shifts", 0)),
            "actor_shifts": int(pc.get("actor_shifts", 0)),
            "disorientation_score": pc.get("disorientation_score", 0),
        }

    rc_res = results.get("referential_cohesion")
    if rc_res:
        pc = rc_res.per_chapter or {}
        report.referential_cohesion = {
            "coherence_score": pc.get("coherence_score", 0),
            "subject_churn": pc.get("subject_churn", 0),
            "entity_count": int(pc.get("entity_count", 0)),
        }

    report_dict = report.to_dict()

    # Carry forward non-metric fields from previous report
    if metrics_only and previous_report:
        for key in ("dialogue", "pacing", "vocabulary_crutches", "pattern_hits"):
            if key in previous_report and key not in report_dict:
                report_dict[key] = previous_report[key]

    # --- Build ProseMetrics from report_dict ---

    ic_dict = report_dict.get("info_contour") or {}
    spikes_val = ic_dict.get("spikes", 0)
    if isinstance(spikes_val, list):
        spikes_val = len(spikes_val)
    flatlines_val = ic_dict.get("flatlines", 0)
    if isinstance(flatlines_val, list):
        flatlines_val = len(flatlines_val)

    fg_dict = report_dict.get("foregrounding") or {}
    pd_dict = report_dict.get("psychic_distance") or {}

    dr_dict = report_dict.get("discourse_relations") or {}
    cn_dict = report_dict.get("concreteness") or {}
    ss_dict = report_dict.get("situation_shifts") or {}

    # Compute shift rate (shifts per paragraph)
    from prose_doctor.text import split_paragraphs
    n_paras = max(len(split_paragraphs(text)), 1)
    total_shifts = ss_dict.get("total_shifts", 0)

    metrics = ProseMetrics(
        pd_mean=pd_dict.get("mean_distance", 0),
        pd_std=pd_dict.get("std_distance", 0),
        fg_inversion=fg_dict.get("inversion_pct", 0),
        fg_sl_cv=fg_dict.get("sentence_length_cv", 0),
        fg_fragment=fg_dict.get("fragment_pct", 0),
        ic_rhythmicity=ic_dict.get("rhythmicity", 0),
        ic_spikes=spikes_val,
        ic_flatlines=flatlines_val,
        dr_entropy=dr_dict.get("relation_entropy", 0),
        dr_implicit=dr_dict.get("implicit_ratio", 1.0),
        cn_abstract=cn_dict.get("abstractness_ratio", 0),
        ss_shift_rate=round(total_shifts / n_paras, 3),
    )

    return metrics, report_dict
