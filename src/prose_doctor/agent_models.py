"""Pydantic models for the revision agent."""
from __future__ import annotations

from pydantic import BaseModel, computed_field

BASELINES: dict[str, tuple[float, str]] = {
    "pd_mean":       (0.336, "higher"),
    "pd_std":        (0.093, "higher"),
    "fg_inversion":  (44.2,  "higher"),
    "fg_sl_cv":      (0.706, "higher"),
    "fg_fragment":   (6.7,   "lower"),
    "ic_rhythmicity":(0.129, "lower"),
    "ic_spikes":     (7.7,   "higher"),
    "ic_flatlines":  (3.1,   "lower"),
    # Discourse — human prose uses diverse, explicit connectives
    "dr_entropy":    (0.65,  "higher"),   # human range 0.62-0.82, AI mean 0.43
    "dr_implicit":   (0.90,  "lower"),    # human range 0.86-0.90, AI mean 0.94
    # Concreteness — human prose is more abstract than AI
    "cn_abstract":   (0.27,  "higher"),   # human range 0.23-0.34, AI mean 0.19
    # Situation shifts — normalized to per-paragraph rate
    "ss_shift_rate": (1.5,   "higher"),   # human ~2 shifts/para, AI ~1 shift/para
}


class ProseMetrics(BaseModel):
    """Structured metrics from a deep scan."""
    pd_mean: float
    pd_std: float
    fg_inversion: float
    fg_sl_cv: float
    fg_fragment: float
    ic_rhythmicity: float
    ic_spikes: int
    ic_flatlines: int
    dr_entropy: float = 0.0
    dr_implicit: float = 1.0
    cn_abstract: float = 0.0
    ss_shift_rate: float = 0.0

    def _metric_distance(self, key: str) -> float:
        """Normalized distance from baseline for one metric. 0 = at/past baseline."""
        baseline, direction = BASELINES[key]
        value = getattr(self, key)
        if baseline == 0:
            return 0.0
        if direction == "higher":
            gap = (baseline - value) / baseline
        else:
            gap = (value - baseline) / baseline
        return max(0.0, gap)

    @computed_field
    @property
    def total_distance(self) -> float:
        return round(sum(self._metric_distance(k) for k in BASELINES), 4)

    @computed_field
    @property
    def worst_metric(self) -> str:
        return max(BASELINES, key=lambda k: self._metric_distance(k))

    def distances(self) -> dict[str, float]:
        return {k: round(self._metric_distance(k), 4) for k in BASELINES}


class EditResult(BaseModel):
    """Result of a replace_passage call."""
    accepted: bool
    reason: str
    metrics_before: ProseMetrics
    metrics_after: ProseMetrics | None = None


class RevisionResult(BaseModel):
    """Final output of the revision agent."""
    final_text: str
    metrics_initial: ProseMetrics
    metrics_final: ProseMetrics
    turns_used: int
    edits_accepted: int
    edits_rejected: int
    metrics_improved: list[str]
    metrics_worsened: list[str]
