from __future__ import annotations

from pathlib import Path

import pytest

from prose_doctor.critique_config import CritiqueConfig, METRIC_LABELS

EXPECTED_BASELINE_KEYS = {
    "pd_mean",
    "pd_std",
    "fg_inversion",
    "fg_sl_cv",
    "fg_fragment",
    "ic_rhythmicity",
    "ic_spikes",
    "ic_flatlines",
    "dr_entropy",
    "dr_implicit",
    "cn_abstract",
    "ss_shift_rate",
}

EXPECTED_BASELINES = {
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


def test_default_baselines_match_current():
    cfg = CritiqueConfig()
    assert len(cfg.baselines) == 12
    assert set(cfg.baselines.keys()) == EXPECTED_BASELINE_KEYS
    for key, (expected_val, expected_dir) in EXPECTED_BASELINES.items():
        actual_val, actual_dir = cfg.baselines[key]
        assert actual_val == pytest.approx(expected_val), f"{key} value mismatch"
        assert actual_dir == expected_dir, f"{key} direction mismatch"


def test_default_prescriptions_exist_for_all_baselines():
    cfg = CritiqueConfig()
    for key in cfg.baselines:
        assert key in cfg.prescriptions, f"Missing prescription for {key}"
        assert isinstance(cfg.prescriptions[key], str), f"Prescription for {key} is not a str"
        assert len(cfg.prescriptions[key]) > 0, f"Prescription for {key} is empty"


def test_metric_labels_cover_all_baselines():
    cfg = CritiqueConfig()
    for key in cfg.baselines:
        assert key in METRIC_LABELS, f"Missing METRIC_LABELS entry for {key}"


def test_yaml_round_trip(tmp_path: Path):
    cfg = CritiqueConfig(name="test-config", parent="default")
    out_path = tmp_path / "sub" / "critique.yaml"
    cfg.to_yaml(out_path)
    assert out_path.exists()

    loaded = CritiqueConfig.from_yaml(out_path)

    assert loaded.name == cfg.name
    assert loaded.parent == cfg.parent
    assert loaded.major_threshold == pytest.approx(cfg.major_threshold)
    assert loaded.minor_threshold == pytest.approx(cfg.minor_threshold)
    assert loaded.strength_threshold == pytest.approx(cfg.strength_threshold)
    assert loaded.max_turns == cfg.max_turns
    assert loaded.regression_limit == pytest.approx(cfg.regression_limit)
    assert loaded.temperature == pytest.approx(cfg.temperature)

    assert set(loaded.baselines.keys()) == set(cfg.baselines.keys())
    for key in cfg.baselines:
        orig_val, orig_dir = cfg.baselines[key]
        loaded_val, loaded_dir = loaded.baselines[key]
        assert loaded_val == pytest.approx(orig_val), f"{key} value after round-trip"
        assert loaded_dir == orig_dir, f"{key} direction after round-trip"

    assert loaded.prescriptions == cfg.prescriptions


def test_default_severity_gates():
    cfg = CritiqueConfig()
    assert cfg.major_threshold == pytest.approx(0.25)
    assert cfg.minor_threshold == pytest.approx(0.15)
    assert cfg.strength_threshold == pytest.approx(0.05)


def test_default_revision_loop_params():
    cfg = CritiqueConfig()
    assert cfg.max_turns == 8
    assert cfg.regression_limit == pytest.approx(0.20)
    assert cfg.temperature == pytest.approx(0.7)


def test_default_issue_finder_thresholds():
    cfg = CritiqueConfig()
    # discourse
    assert cfg.discourse_entropy_gate == pytest.approx(0.55)
    assert cfg.discourse_implicit_gate == pytest.approx(0.92)
    assert cfg.consecutive_implicit_trigger == 3
    assert cfg.additive_count_trigger == 2
    # concreteness
    assert cfg.concrete_run_trigger == 4
    assert cfg.concrete_para_mean_threshold == pytest.approx(3.2)
    assert cfg.abstract_ratio_gate == pytest.approx(0.15)
    assert cfg.vague_density_gate == pytest.approx(0.5)
    # shifts
    assert cfg.shift_rate_gate == pytest.approx(1.2)
    assert cfg.no_shift_run_trigger == 5
    # psychic distance
    assert cfg.pd_baseline_margin == pytest.approx(0.05)
    assert cfg.pd_cause_threshold == 2
    # inversions
    assert cfg.inversion_pct_gate == pytest.approx(15.0)
    # spikes
    assert cfg.spike_surprisal_margin == pytest.approx(0.1)
