"""Tests for config loading."""

import tempfile
from pathlib import Path

from prose_doctor.config import ProjectConfig


def test_default_config():
    cfg = ProjectConfig()
    assert cfg.character_names == []
    assert cfg.exempt_words == set()
    assert cfg.pov_map == {}


def test_load_from_toml():
    toml_content = """\
[prose-doctor]
character_names = ["Rook", "Cassian"]
exempt_words = ["rook", "cassian"]

[prose-doctor.pov]
Rook = ["chapter_01", "chapter_04"]

[prose-doctor.density_budgets]
tricolon = 3

[prose-doctor.models]
slop_classifier = "my-org/slop-model"
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / ".prose-doctor.toml"
        p.write_text(toml_content)
        cfg = ProjectConfig.load(Path(tmpdir))

    assert cfg.character_names == ["Rook", "Cassian"]
    assert "rook" in cfg.exempt_words
    assert cfg.pov_map["Rook"] == ["chapter_01", "chapter_04"]
    assert cfg.density_budgets["tricolon"] == 3
    assert cfg.slop_classifier_model == "my-org/slop-model"


def test_load_missing_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ProjectConfig.load(Path(tmpdir))
    assert cfg.character_names == []


def test_default_template():
    template = ProjectConfig.default_template()
    assert "[prose-doctor]" in template
    assert "character_names" in template
