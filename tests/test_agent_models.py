from prose_doctor.agent_models import ProseMetrics, EditResult, RevisionResult, BASELINES


def test_prose_metrics_total_distance_at_baseline():
    """Total distance is 0 when all metrics are at baseline."""
    m = ProseMetrics(
        pd_mean=0.336, pd_std=0.093, fg_inversion=44.2,
        fg_sl_cv=0.706, fg_fragment=6.7, ic_rhythmicity=0.129,
        ic_spikes=8, ic_flatlines=3,
        dr_entropy=0.65, dr_implicit=0.90, cn_abstract=0.27, ss_shift_rate=1.5,
    )
    assert m.total_distance == 0.0


def test_prose_metrics_total_distance_above_baseline():
    """Total distance is 0 when metrics exceed baseline in the right direction."""
    m = ProseMetrics(
        pd_mean=0.5, pd_std=0.15, fg_inversion=50.0,
        fg_sl_cv=0.8, fg_fragment=4.0, ic_rhythmicity=0.08,
        ic_spikes=10, ic_flatlines=2,
        dr_entropy=0.70, dr_implicit=0.85, cn_abstract=0.30, ss_shift_rate=2.0,
    )
    assert m.total_distance == 0.0


def test_prose_metrics_worst_metric():
    """worst_metric returns the metric farthest from baseline."""
    m = ProseMetrics(
        pd_mean=0.1, pd_std=0.093, fg_inversion=44.2,
        fg_sl_cv=0.706, fg_fragment=6.7, ic_rhythmicity=0.129,
        ic_spikes=7, ic_flatlines=3,
        dr_entropy=0.65, dr_implicit=0.90, cn_abstract=0.27, ss_shift_rate=1.5,
    )
    assert m.worst_metric == "pd_mean"


def test_prose_metrics_distances():
    m = ProseMetrics(
        pd_mean=0.2, pd_std=0.093, fg_inversion=44.2,
        fg_sl_cv=0.706, fg_fragment=6.7, ic_rhythmicity=0.129,
        ic_spikes=7, ic_flatlines=3,
        dr_entropy=0.65, dr_implicit=0.90, cn_abstract=0.27, ss_shift_rate=1.5,
    )
    d = m.distances()
    assert d["pd_mean"] > 0
    assert d["pd_std"] == 0
    assert d["fg_inversion"] == 0
    assert d["dr_entropy"] == 0
    assert d["cn_abstract"] == 0


def test_edit_result_serialization():
    before = ProseMetrics(
        pd_mean=0.2, pd_std=0.05, fg_inversion=30.0,
        fg_sl_cv=0.5, fg_fragment=10.0, ic_rhythmicity=0.2,
        ic_spikes=3, ic_flatlines=5,
    )
    er = EditResult(
        accepted=True, reason="improved pd_mean by 0.05",
        metrics_before=before, metrics_after=before,
    )
    d = er.model_dump()
    assert d["accepted"] is True
    assert "metrics_before" in d


def test_prose_metrics_custom_baselines():
    """ProseMetrics uses custom baselines when provided."""
    custom = {"pd_mean": (0.5, "higher"), "pd_std": (0.1, "higher")}
    m = ProseMetrics(
        pd_mean=0.5, pd_std=0.1,
        fg_inversion=44.2, fg_sl_cv=0.706, fg_fragment=6.7,
        ic_rhythmicity=0.129, ic_spikes=8, ic_flatlines=3,
        baselines=custom,
    )
    assert m.total_distance == 0.0
    assert len(m.distances()) == 2


def test_revision_result_serialization():
    m = ProseMetrics(
        pd_mean=0.336, pd_std=0.093, fg_inversion=44.2,
        fg_sl_cv=0.706, fg_fragment=6.7, ic_rhythmicity=0.129,
        ic_spikes=7, ic_flatlines=3,
    )
    r = RevisionResult(
        final_text="revised text",
        metrics_initial=m, metrics_final=m,
        turns_used=3, edits_accepted=2, edits_rejected=1,
        metrics_improved=["pd_mean"], metrics_worsened=[],
    )
    d = r.model_dump()
    assert d["turns_used"] == 3
    assert d["final_text"] == "revised text"
