from prose_doctor.validation.discriminator import compute_discrimination
from prose_doctor.validation.promotion import check_tier


def test_compute_discrimination():
    human_scores = [0.4, 0.35, 0.38, 0.42, 0.39]
    llm_scores = [0.2, 0.18, 0.22, 0.19, 0.21]
    result = compute_discrimination(human_scores, llm_scores)
    assert result["cohens_d"] > 0.5
    assert result["p_value"] < 0.05


def test_check_tier_experimental():
    stats = {"cohens_d": 0.2, "p_value": 0.3}
    assert check_tier(stats, revision_evidence=[]) == "experimental"


def test_check_tier_validated():
    stats = {"cohens_d": 0.8, "p_value": 0.001}
    assert check_tier(stats, revision_evidence=[]) == "validated"


def test_check_tier_stable():
    stats = {"cohens_d": 0.8, "p_value": 0.001}
    evidence = ["b1ch01:3_accepted", "b2ch06:2_accepted"]
    assert check_tier(stats, revision_evidence=evidence) == "stable"


def test_check_tier_stable_requires_distinct_chapters():
    stats = {"cohens_d": 0.8, "p_value": 0.001}
    evidence = ["b1ch01:3_accepted", "b1ch01:2_accepted"]  # same chapter
    assert check_tier(stats, revision_evidence=evidence) == "validated"
