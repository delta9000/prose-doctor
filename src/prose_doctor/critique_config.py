from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


METRIC_LABELS: dict[str, str] = {
    "pd_mean":        "Psychic Distance (mean)",
    "pd_std":         "Psychic Distance (variability)",
    "fg_inversion":   "Foregrounding: Syntactic Inversions (%)",
    "fg_sl_cv":       "Foregrounding: Sentence-Length Variation (CV)",
    "fg_fragment":    "Foregrounding: Fragment Rate (%)",
    "ic_rhythmicity": "Info Contour: Rhythmicity",
    "ic_spikes":      "Info Contour: Surprisal Spikes (%)",
    "ic_flatlines":   "Info Contour: Flatlines (%)",
    "dr_entropy":     "Discourse Relations: Entropy",
    "dr_implicit":    "Discourse Relations: Implicit Rate",
    "cn_abstract":    "Concreteness: Abstract Ratio",
    "ss_shift_rate":  "Situation Shifts: Shift Rate",
}

_DEFAULT_BASELINES: dict[str, tuple[float, str]] = {
    "pd_mean":        (0.336, "higher"),
    "pd_std":         (0.093, "higher"),
    "fg_inversion":   (44.2,  "higher"),
    "fg_sl_cv":       (0.706, "higher"),
    "fg_fragment":    (6.7,   "lower"),
    "ic_rhythmicity": (0.129, "lower"),
    "ic_spikes":      (7.7,   "higher"),
    "ic_flatlines":   (3.1,   "lower"),
    "dr_entropy":     (0.65,  "higher"),
    "dr_implicit":    (0.90,  "lower"),
    "cn_abstract":    (0.27,  "higher"),
    "ss_shift_rate":  (1.5,   "higher"),
}

_DEFAULT_PRESCRIPTIONS: dict[str, str] = {
    "pd_mean": (
        "You're narrating from outside the character. Pick the 3–5 most emotionally "
        "charged moments and rewrite them from deep inside the character's perception: "
        "filter sensory detail through their emotional state, let their assumptions and "
        "misreadings color what they notice, and replace neutral description with "
        "subjective interpretation. The reader should feel the scene through the "
        "character's nervous system, not observe it from a distance."
    ),
    "pd_std": (
        "The psychic distance stays flat throughout the passage. Vary the narrative "
        "intimacy deliberately: pull back to wide-angle observation at scene transitions "
        "or moments of shock, then plunge close during peak emotion or decision. A "
        "distance shift of even one register creates rhythm and emphasis. Find two or "
        "three places to consciously move the camera closer or farther."
    ),
    "fg_inversion": (
        "Too many sentences follow subject-verb-object order. Restructure 5–8 sentences "
        "to lead with a prepositional phrase, a participial phrase, an adverbial clause, "
        "or the object instead of the subject. Front-loading non-subject material "
        "creates emphasis, varies cadence, and makes prose feel less mechanical. "
        "Prioritize sentences at emotional peaks or scene transitions."
    ),
    "fg_sl_cv": (
        "Sentence lengths are too uniform — the prose has a flat, metronomic rhythm. "
        "Introduce more variation: follow a long, complex sentence with a short, blunt "
        "one. Let tension build in short bursts. Let description sprawl when the scene "
        "calls for it. Aim for peaks and valleys rather than a plateau."
    ),
    "fg_fragment": (
        "Some short fragments here don't earn their weight. NOT EVERY FRAGMENT IS BAD — "
        "fragments that land a punch (visceral detail, tonal shift, rhythmic break after "
        "a long sentence) should stay. Only merge fragments that are vague, generic, or "
        "add nothing the surrounding sentences don't already convey. When merging, fold "
        "the fragment's content into the adjacent sentence naturally — don't just glue "
        "them with a comma. Leave strong fragments alone."
    ),
    "ic_rhythmicity": (
        "The information density is too regular — every sentence carries roughly the "
        "same cognitive load, creating a droning effect. Break the pattern: compress "
        "routine action into a single clause, then expand the emotionally significant "
        "moment with layered detail. Let the reader's attention accelerate and decelerate "
        "rather than cruise at a fixed speed."
    ),
    "ic_spikes": (
        "The passage lacks surprise — information arrives at a predictable rate with no "
        "spikes of unexpected detail or word choice. Find 4–6 places to introduce "
        "something the reader couldn't have predicted: an unusual word, an unexpected "
        "comparison, a detail that reframes what came before. Surprisal creates "
        "engagement; without it prose reads as merely competent."
    ),
    "ic_flatlines": (
        "Too many sentences add no new information — they restate, echo, or pad. Cut or "
        "compress any sentence whose content is already implied by what surrounds it. "
        "If a sentence exists only to smooth a transition or signal an emotion already "
        "shown through action, remove it. Every sentence should move something forward: "
        "plot, character, image, or tone."
    ),
    "dr_entropy": (
        "The passage relies almost entirely on additive connectives ('and', 'then', "
        "'also'). Replace some with causal ('because', 'so', 'therefore'), adversative "
        "('but', 'although', 'yet'), or conditional ('if', 'unless') relations. Varying "
        "discourse relation types shows how ideas relate rather than merely listing them, "
        "and creates a richer logical texture."
    ),
    "dr_implicit": (
        "Too many discourse relations are left implicit — the logical or emotional links "
        "between sentences are unmarked, leaving the reader to infer connections that "
        "should be felt. Add connective tissue: a 'but' before the reversal, a 'because' "
        "before the explanation, a 'so' before the consequence. Making more relations "
        "explicit clarifies causality and emotional logic without overexplaining."
    ),
    "cn_abstract": (
        "The passage sits on the sensory surface without reflection or interpretation. "
        "Add moments where the point-of-view character draws a conclusion, makes an "
        "association, voices an opinion, or revisits a memory triggered by what they "
        "perceive. Abstract thought anchored to concrete sensation creates the "
        "impression of a mind at work — not just a camera recording events."
    ),
    "ss_shift_rate": (
        "The scene is static — no new characters enter, no location changes, no time "
        "passes, and no topic shifts mark the passage. Break it up: add a time reference "
        "('ten minutes later'), move to a new physical position, introduce a new object "
        "of attention, or bring someone into or out of the scene. Situation shifts "
        "signal progress and prevent the reader from feeling trapped in amber."
    ),
}


@dataclass
class CritiqueConfig:
    # Identity
    name: str = "default"
    parent: str | None = None

    # Baselines: metric_key -> (target_value, "higher"|"lower")
    baselines: dict[str, tuple[float, str]] = field(
        default_factory=lambda: dict(_DEFAULT_BASELINES)
    )

    # Severity gates
    major_threshold: float = 0.25
    minor_threshold: float = 0.15
    strength_threshold: float = 0.05

    # Issue-finder thresholds — discourse
    discourse_entropy_gate: float = 0.55
    discourse_implicit_gate: float = 0.92
    consecutive_implicit_trigger: int = 3
    additive_count_trigger: int = 2

    # Issue-finder thresholds — concreteness
    concrete_run_trigger: int = 4
    concrete_para_mean_threshold: float = 3.2
    abstract_ratio_gate: float = 0.15
    vague_density_gate: float = 0.5

    # Issue-finder thresholds — situation shifts
    shift_rate_gate: float = 1.2
    no_shift_run_trigger: int = 5

    # Issue-finder thresholds — psychic distance
    pd_baseline_margin: float = 0.05
    pd_cause_threshold: int = 2

    # Issue-finder thresholds — inversions
    inversion_pct_gate: float = 15.0

    # Issue-finder thresholds — surprisal spikes
    spike_surprisal_margin: float = 0.1

    # Prescriptions: one revision instruction per baseline metric key
    prescriptions: dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_PRESCRIPTIONS)
    )

    # Revision loop
    max_turns: int = 8
    regression_limit: float = 0.20
    temperature: float = 0.7

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_yaml(self, path: Path) -> None:
        """Save this config to a YAML file, creating parent directories as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Baselines are stored as lists (YAML has no tuple type)
        data: dict[str, Any] = {
            "name": self.name,
            "parent": self.parent,
            "baselines": {k: list(v) for k, v in self.baselines.items()},
            "major_threshold": self.major_threshold,
            "minor_threshold": self.minor_threshold,
            "strength_threshold": self.strength_threshold,
            "discourse_entropy_gate": self.discourse_entropy_gate,
            "discourse_implicit_gate": self.discourse_implicit_gate,
            "consecutive_implicit_trigger": self.consecutive_implicit_trigger,
            "additive_count_trigger": self.additive_count_trigger,
            "concrete_run_trigger": self.concrete_run_trigger,
            "concrete_para_mean_threshold": self.concrete_para_mean_threshold,
            "abstract_ratio_gate": self.abstract_ratio_gate,
            "vague_density_gate": self.vague_density_gate,
            "shift_rate_gate": self.shift_rate_gate,
            "no_shift_run_trigger": self.no_shift_run_trigger,
            "pd_baseline_margin": self.pd_baseline_margin,
            "pd_cause_threshold": self.pd_cause_threshold,
            "inversion_pct_gate": self.inversion_pct_gate,
            "spike_surprisal_margin": self.spike_surprisal_margin,
            "prescriptions": self.prescriptions,
            "max_turns": self.max_turns,
            "regression_limit": self.regression_limit,
            "temperature": self.temperature,
        }
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))

    @classmethod
    def from_yaml(cls, path: Path) -> CritiqueConfig:
        """Load a CritiqueConfig from a YAML file."""
        path = Path(path)
        data: dict[str, Any] = yaml.safe_load(path.read_text())

        # Convert baselines from list -> tuple
        raw_baselines = data.pop("baselines", {})
        baselines: dict[str, tuple[float, str]] = {
            k: (float(v[0]), str(v[1])) for k, v in raw_baselines.items()
        }

        return cls(baselines=baselines, **data)
