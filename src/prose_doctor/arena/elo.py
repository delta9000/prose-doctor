from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

K = 32


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))


@dataclass
class EloTracker:
    ratings: dict[str, float] = field(default_factory=dict)
    matches: list[dict] = field(default_factory=list)

    def add_config(self, name: str, rating: float = 1500.0) -> None:
        if name not in self.ratings:
            self.ratings[name] = rating

    def rating(self, name: str) -> float:
        return self.ratings[name]

    def record_match(
        self,
        config_a: str,
        config_b: str,
        winner: str,
        story_id: str,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        ra = self.ratings[config_a]
        rb = self.ratings[config_b]

        ea = _expected(ra, rb)
        eb = _expected(rb, ra)

        if winner == config_a:
            sa, sb = 1.0, 0.0
        elif winner == config_b:
            sa, sb = 0.0, 1.0
        elif winner == "tie":
            sa, sb = 0.5, 0.5
        else:
            raise ValueError(f"winner must be config_a, config_b, or 'tie', got {winner!r}")

        new_ra = ra + K * (sa - ea)
        new_rb = rb + K * (sb - eb)

        self.ratings[config_a] = new_ra
        self.ratings[config_b] = new_rb

        record: dict[str, Any] = {
            "config_a": config_a,
            "config_b": config_b,
            "winner": winner,
            "story_id": story_id,
            "reason": reason,
            "rating_a_after": new_ra,
            "rating_b_after": new_rb,
        }
        if metadata:
            record.update(metadata)

        self.matches.append(record)

    def leaderboard(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def save(self, matches_path: Path | str, ratings_path: Path | str) -> None:
        matches_path = Path(matches_path)
        ratings_path = Path(ratings_path)

        with matches_path.open("a", encoding="utf-8") as f:
            for match in self.matches:
                f.write(json.dumps(match) + "\n")

        ratings_data = {"configs": {name: {"elo": elo} for name, elo in self.ratings.items()}}
        ratings_path.write_text(json.dumps(ratings_data, indent=2), encoding="utf-8")

        self.matches.clear()

    @classmethod
    def load(cls, matches_path: Path | str, ratings_path: Path | str) -> EloTracker:
        ratings_path = Path(ratings_path)
        ratings_data = json.loads(ratings_path.read_text(encoding="utf-8"))
        ratings = {name: cfg["elo"] for name, cfg in ratings_data["configs"].items()}
        return cls(ratings=ratings)
