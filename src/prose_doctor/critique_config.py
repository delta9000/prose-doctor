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
        "The psychic distance is too far — the prose reads like a report, not lived "
        "experience. Move closer to the character's perception.\n\n"
        "EXAMPLE:\n"
        "  Before: 'The room was cold. She noticed the window was open.'\n"
        "  After:  'Cold air bit her arms. The window — who had left it open?'\n"
        "  Why: Sensation ('bit her arms') replaces observation ('was cold'). "
        "Question shows the character's mind, not the narrator's.\n\n"
        "KEEP AS-IS when the distance is deliberate (establishing shots, time skips, "
        "moments of shock where the character dissociates)."
    ),
    "pd_std": (
        "The psychic distance stays flat. Vary it: pull back at scene transitions, "
        "push close during peak emotion.\n\n"
        "EXAMPLE:\n"
        "  Before: 'He walked into the bar. He felt nervous. He ordered a drink.'\n"
        "  After:  'The bar was half-empty, neon pooling on wet tables. His stomach "
        "clenched as he slid onto the stool. \"Whiskey.\"'\n"
        "  Why: Wide shot (neon, tables) then close interiority (stomach) then "
        "dialogue. Three registers in three sentences."
    ),
    "fg_inversion": (
        "Too many subject-verb-object sentences. Restructure 2-3 where a different "
        "opening creates natural emphasis — not as a mechanical pattern.\n\n"
        "EXAMPLE:\n"
        "  Before: 'Marcus walked down the corridor. He checked each door.'\n"
        "  After:  'Down the corridor Marcus walked, checking each door.'\n"
        "  Why: Prepositional opening varies rhythm. Only do this where the "
        "fronted element deserves emphasis.\n\n"
        "KEEP AS-IS: 'She ran.' (Fragment at a climactic beat — inversion would kill it.)"
    ),
    "fg_sl_cv": (
        "Sentence lengths are too uniform. Mix long flowing sentences with short punches.\n\n"
        "EXAMPLE:\n"
        "  Before: 'He opened the door. He stepped inside. He looked around. "
        "The room was empty.'\n"
        "  After:  'He opened the door and stepped inside, scanning the room "
        "in one quick sweep. Empty.'\n"
        "  Why: Routine action compressed into one sentence. The discovery gets "
        "its own fragment."
    ),
    "fg_fragment": (
        "Some fragments don't earn their weight. NOT EVERY FRAGMENT IS BAD.\n\n"
        "FIX THIS:\n"
        "  Before: 'She paused. The hallway stretched ahead. Dark. Empty.'\n"
        "  After:  'She paused at the mouth of the hallway — dark, empty, "
        "stretching ahead.'\n"
        "  Why: 'Dark. Empty.' are vague. Folded into context naturally.\n\n"
        "KEEP THIS:\n"
        "  'The gun fired. Smoke. Silence. Then screaming.'\n"
        "  Why: Each fragment marks a distinct sensory beat. The rhythm IS the content.\n\n"
        "Rule: if a fragment follows a long sentence and carries a visceral detail "
        "or tonal shift, leave it. If it's generic padding, fold it in."
    ),
    "ic_rhythmicity": (
        "Information density is too regular — every sentence carries the same weight.\n\n"
        "EXAMPLE:\n"
        "  Before: 'He packed his bag. He checked the map. He counted his money. "
        "He closed the door.'\n"
        "  After:  'He packed the bag, checked the map, counted what was left of "
        "the money — forty-three dollars and change, not enough for the bus.'\n"
        "  Why: Routine actions compressed. The money detail expands because it "
        "matters emotionally."
    ),
    "ic_spikes": (
        "The prose is too predictable. Inject surprise at 2-3 key moments.\n\n"
        "EXAMPLE:\n"
        "  Before: 'The old house was quiet. The paint was peeling.'\n"
        "  After:  'The house held its breath. Paint curled from the siding "
        "like dead skin.'\n"
        "  Why: 'Held its breath' personifies unexpectedly. 'Dead skin' reframes "
        "a mundane detail. Both spike the reader's attention."
    ),
    "ic_flatlines": (
        "Sentences are restating or padding. Cut what's already implied.\n\n"
        "EXAMPLE:\n"
        "  Before: 'She was angry. Her fists clenched. She felt a surge of rage.'\n"
        "  After:  'Her fists clenched.'\n"
        "  Why: The fists show anger. The label ('angry') and restatement "
        "('surge of rage') add nothing."
    ),
    "dr_entropy": (
        "Too many additive connectives ('and', 'then'). Show logical relationships.\n\n"
        "EXAMPLE:\n"
        "  Before: 'She opened the door. And the hallway was dark. And she stepped inside.'\n"
        "  After:  'She opened the door, but the hallway beyond was dark. She stepped "
        "inside anyway.'\n"
        "  Why: 'But' creates contrast (expectation vs reality). 'Anyway' shows "
        "decision despite the obstacle. Richer than three 'and's."
    ),
    "dr_implicit": (
        "Logical links between sentences are unmarked. Add connective tissue.\n\n"
        "EXAMPLE:\n"
        "  Before: 'The bridge was out. They took the long way around.'\n"
        "  After:  'The bridge was out, so they took the long way around.'\n"
        "  Why: 'So' makes the causal link explicit. The reader shouldn't have to "
        "infer basic cause and effect.\n\n"
        "Don't overdo it — not every sentence pair needs a connective. Only add "
        "them where the logical relationship is non-obvious."
    ),
    "cn_abstract": (
        "The passage is all surface — no reflection or interpretation.\n\n"
        "EXAMPLE:\n"
        "  Before: 'The coffee was cold. The mug had a chip in the rim.'\n"
        "  After:  'The coffee had gone cold — how long had she been sitting here? "
        "The chip in the rim was old, from the move. Another thing she hadn't fixed.'\n"
        "  Why: Same objects, but now the character's mind is working. The chip "
        "triggers a memory and a judgment. Concrete detail anchors abstract thought."
    ),
    "ss_shift_rate": (
        "The scene is static — nothing changes. Add a shift.\n\n"
        "EXAMPLE:\n"
        "  Before: 'He sat at the desk. He read the file. He turned the page. "
        "He read more.'\n"
        "  After:  'He sat at the desk reading the file until the light changed — "
        "late afternoon now, the shadows longer. He turned the page.'\n"
        "  Why: Time reference ('light changed') signals progress. The reader "
        "feels time passing, not looping."
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
