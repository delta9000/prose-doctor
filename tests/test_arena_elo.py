import json

import pytest

from prose_doctor.arena.elo import EloTracker


def test_elo_winner_gains_rating():
    tracker = EloTracker()
    tracker.add_config("a")
    tracker.add_config("b")
    tracker.record_match("a", "b", winner="a", story_id="s1", reason="better")
    assert tracker.rating("a") > 1500
    assert tracker.rating("b") < 1500


def test_elo_loser_loses_rating():
    tracker = EloTracker()
    tracker.add_config("a")
    tracker.add_config("b")
    tracker.record_match("a", "b", winner="b", story_id="s1", reason="better")
    assert tracker.rating("b") > 1500
    assert tracker.rating("a") < 1500


def test_elo_tie_splits_points():
    tracker = EloTracker()
    tracker.add_config("a")
    tracker.add_config("b")
    tracker.record_match("a", "b", winner="tie", story_id="s1", reason="equal")
    assert tracker.rating("a") == 1500
    assert tracker.rating("b") == 1500


def test_elo_save_load_roundtrip(tmp_path):
    tracker = EloTracker()
    tracker.add_config("a")
    tracker.add_config("b")
    tracker.record_match("a", "b", winner="a", story_id="s1", reason="test")
    matches_path = tmp_path / "matches.jsonl"
    ratings_path = tmp_path / "ratings.json"
    tracker.save(matches_path, ratings_path)
    loaded = EloTracker.load(matches_path, ratings_path)
    assert abs(loaded.rating("a") - tracker.rating("a")) < 0.01
    assert abs(loaded.rating("b") - tracker.rating("b")) < 0.01


def test_elo_leaderboard_sorted():
    tracker = EloTracker()
    tracker.add_config("a")
    tracker.add_config("b")
    tracker.record_match("a", "b", winner="a", story_id="s1", reason="test")
    board = tracker.leaderboard()
    assert board[0][0] == "a"  # winner first
    assert board[0][1] > board[1][1]


def test_elo_save_appends_matches(tmp_path):
    tracker = EloTracker()
    tracker.add_config("a")
    tracker.add_config("b")
    tracker.record_match("a", "b", winner="a", story_id="s1", reason="first")
    matches_path = tmp_path / "matches.jsonl"
    ratings_path = tmp_path / "ratings.json"
    tracker.save(matches_path, ratings_path)

    tracker.record_match("a", "b", winner="b", story_id="s2", reason="second")
    tracker.save(matches_path, ratings_path)

    lines = matches_path.read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["story_id"] == "s1"
    second = json.loads(lines[1])
    assert second["story_id"] == "s2"


def test_elo_save_clears_match_buffer(tmp_path):
    tracker = EloTracker()
    tracker.add_config("a")
    tracker.add_config("b")
    tracker.record_match("a", "b", winner="a", story_id="s1", reason="test")
    matches_path = tmp_path / "matches.jsonl"
    ratings_path = tmp_path / "ratings.json"
    tracker.save(matches_path, ratings_path)
    assert tracker.matches == []


def test_elo_match_record_format(tmp_path):
    tracker = EloTracker()
    tracker.add_config("a")
    tracker.add_config("b")
    tracker.record_match("a", "b", winner="a", story_id="s1", reason="clearer prose", metadata={"judge": "gpt-4o"})
    record = tracker.matches[0]
    assert record["config_a"] == "a"
    assert record["config_b"] == "b"
    assert record["winner"] == "a"
    assert record["story_id"] == "s1"
    assert record["reason"] == "clearer prose"
    assert "rating_a_after" in record
    assert "rating_b_after" in record
    assert record["judge"] == "gpt-4o"


def test_elo_add_config_no_op_if_exists():
    tracker = EloTracker()
    tracker.add_config("a", rating=1600.0)
    tracker.add_config("a", rating=1200.0)  # should be ignored
    assert tracker.rating("a") == 1600.0


def test_elo_invalid_winner_raises():
    tracker = EloTracker()
    tracker.add_config("a")
    tracker.add_config("b")
    with pytest.raises(ValueError):
        tracker.record_match("a", "b", winner="c", story_id="s1")


def test_elo_ratings_json_format(tmp_path):
    tracker = EloTracker()
    tracker.add_config("a", rating=1520.0)
    tracker.add_config("b", rating=1480.0)
    matches_path = tmp_path / "matches.jsonl"
    ratings_path = tmp_path / "ratings.json"
    tracker.save(matches_path, ratings_path)
    data = json.loads(ratings_path.read_text())
    assert "configs" in data
    assert data["configs"]["a"]["elo"] == 1520.0
    assert data["configs"]["b"]["elo"] == 1480.0
