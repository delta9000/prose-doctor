"""Pydantic models for the revision agent."""
from __future__ import annotations

from pydantic import BaseModel, PrivateAttr, computed_field


def _default_baselines() -> dict[str, tuple[float, str]]:
    from prose_doctor.critique_config import CritiqueConfig
    return CritiqueConfig().baselines


# Backward compat
BASELINES = _default_baselines()


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

    _baselines: dict[str, tuple[float, str]] | None = PrivateAttr(default=None)

    def __init__(self, baselines: dict[str, tuple[float, str]] | None = None, **data):
        super().__init__(**data)
        self._baselines = baselines

    @property
    def baselines(self) -> dict[str, tuple[float, str]]:
        if self._baselines is not None:
            return self._baselines
        return _default_baselines()

    def _metric_distance(self, key: str) -> float:
        """Normalized distance from baseline for one metric. 0 = at/past baseline."""
        baseline, direction = self.baselines[key]
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
        return round(sum(self._metric_distance(k) for k in self.baselines), 4)

    @computed_field
    @property
    def worst_metric(self) -> str:
        return max(self.baselines, key=lambda k: self._metric_distance(k))

    def distances(self) -> dict[str, float]:
        return {k: round(self._metric_distance(k), 4) for k in self.baselines}


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
