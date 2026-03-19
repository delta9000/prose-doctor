"""Core analyzer smoke tests."""

from prose_doctor.analyzers.doctor import diagnose
from prose_doctor.analyzers.proof_scanner import ProofScanner
from prose_doctor.analyzers.vocabulary import find_vocabulary_crutches
from prose_doctor.analyzers.density import DensityAnalyzer


def test_proof_scanner(sample_chapter):
    scanner = ProofScanner()
    findings = scanner.scan(sample_chapter)
    # Should find some patterns in the sample
    assert isinstance(findings, list)
    # Sample has "said quietly" which is an adverb dialogue tag
    categories = [f.category for f in findings]
    assert "adverb_dialogue_tag" in categories


def test_proof_scanner_with_names(sample_chapter):
    scanner = ProofScanner(character_names=["Rook", "Cassian"])
    findings = scanner.scan(sample_chapter)
    assert isinstance(findings, list)


def test_vocabulary_crutches():
    # Create text with a deliberately overused word
    text = "The steady rhythm was steady and the steady beat continued. " * 20
    crutches = find_vocabulary_crutches(text, threshold_per_1k=1.0)
    words = [c["word"] for c in crutches]
    assert "steady" in words


def test_vocabulary_crutches_short_text():
    result = find_vocabulary_crutches("Too short.")
    assert result == []


def test_density_analyzer(sample_chapter):
    da = DensityAnalyzer()
    report = da.analyze(sample_chapter, filename="test.md")
    assert report.filename == "test.md"
    assert report.word_count > 0
    assert isinstance(report.hits, list)


def test_diagnose(sample_chapter):
    health = diagnose(sample_chapter, filename="chapter_01.md")
    assert health.filename == "chapter_01.md"
    assert health.word_count > 0
    assert isinstance(health.total_issues, int)


def test_diagnose_to_dict(sample_chapter):
    health = diagnose(sample_chapter, filename="test.md")
    d = health.to_dict()
    assert d["filename"] == "test.md"
    assert "total_issues" in d
    assert "vocabulary_crutches" in d
