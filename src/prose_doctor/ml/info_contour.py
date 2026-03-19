"""Information contour analyzer: surprisal rhythm detection.

Extends GPT-2 perplexity scoring into a structural analysis of how
information density oscillates across a chapter. Based on Tsipidi & Nowak
(EMNLP 2024) showing information rate is not uniform but follows structured
contours shaped by discourse hierarchy.

Detects:
  - Information flatlines (uniform surprisal = LLM fingerprint)
  - Surprisal spikes without narrative payoff
  - Dominant information rhythm (natural scene cycle length)
  - Rhythmicity score (structured oscillation vs flat vs chaotic)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

from prose_doctor.text import split_paragraphs_with_breaks


@dataclass
class InfoContourResult:
    """Information contour analysis for a chapter."""

    filename: str
    sentence_count: int
    sentence_surprisals: list[float]

    # Summary stats
    mean_surprisal: float
    std_surprisal: float
    cv_surprisal: float  # coefficient of variation

    # Rhythm analysis
    dominant_period: int  # sentences per dominant cycle
    dominant_period_words: int  # approximate words per cycle
    rhythmicity: float  # 0-1, how periodic is the contour
    spectral_entropy: float  # high = chaotic, low = structured

    # Flags
    flatlines: list[dict] = field(default_factory=list)  # {start, end, length, mean}
    spikes: list[dict] = field(default_factory=list)  # {index, surprisal, text}

    @property
    def label(self) -> str:
        if self.rhythmicity > 0.6:
            return "strongly rhythmic"
        elif self.rhythmicity > 0.35:
            return "moderately rhythmic"
        elif self.cv_surprisal < 0.15:
            return "flat (possible LLM fingerprint)"
        else:
            return "chaotic"


def _sentence_surprisal(text: str, model, tokenizer, device: str) -> float:
    """Compute per-sentence surprisal (negative log likelihood) using GPT-2."""
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    return outputs.loss.item()  # mean NLL per token


def _analyze_spectrum(surprisals: list[float]) -> tuple[int, float, float]:
    """FFT analysis of the surprisal contour.

    Returns (dominant_period, rhythmicity, spectral_entropy).
    """
    import numpy as np

    n = len(surprisals)
    if n < 8:
        return n, 0.0, 1.0

    # Detrend (remove linear trend)
    x = np.array(surprisals)
    t = np.arange(n, dtype=float)
    slope = np.polyfit(t, x, 1)
    detrended = x - np.polyval(slope, t)

    # Hann window to reduce spectral leakage
    windowed = detrended * np.hanning(n)

    # FFT
    fft = np.fft.rfft(windowed)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n)

    # Skip DC component
    power = power[1:]
    freqs = freqs[1:]

    if len(power) == 0 or power.sum() == 0:
        return n, 0.0, 1.0

    # Dominant frequency (highest power, excluding very low frequencies)
    # Skip the first few bins to avoid catching the whole-chapter trend
    min_bin = max(1, len(power) // 20)
    search_power = power[min_bin:]
    search_freqs = freqs[min_bin:]

    if len(search_power) == 0:
        return n, 0.0, 1.0

    peak_idx = np.argmax(search_power) + min_bin
    dominant_freq = freqs[peak_idx]
    dominant_period = int(1.0 / dominant_freq) if dominant_freq > 0 else n

    # Rhythmicity: how much of the total power is in the top 3 frequencies
    sorted_power = np.sort(power)[::-1]
    total_power = power.sum()
    top3_power = sorted_power[:3].sum()
    rhythmicity = float(top3_power / total_power) if total_power > 0 else 0.0

    # Spectral entropy (normalized)
    p = power / total_power
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log2(p)))
    max_entropy = np.log2(len(power)) if len(power) > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

    return dominant_period, rhythmicity, normalized_entropy


def _detect_flatlines(
    surprisals: list[float],
    window: int = 8,
    cv_threshold: float = 0.08,
) -> list[dict]:
    """Detect stretches where surprisal barely varies (information flatlines)."""
    import numpy as np

    flatlines = []
    i = 0
    while i <= len(surprisals) - window:
        chunk = surprisals[i : i + window]
        mean = np.mean(chunk)
        std = np.std(chunk)
        cv = std / mean if mean > 0 else 0

        if cv < cv_threshold:
            # Extend the flatline as far as it goes
            end = i + window
            while end < len(surprisals):
                extended = surprisals[i : end + 1]
                if np.std(extended) / np.mean(extended) > cv_threshold * 1.5:
                    break
                end += 1

            flatlines.append({
                "start": i,
                "end": end - 1,
                "length": end - i,
                "mean_surprisal": round(float(np.mean(surprisals[i:end])), 3),
            })
            i = end
        else:
            i += 1

    return flatlines


def _detect_spikes(
    surprisals: list[float],
    sentences: list[str],
    z_threshold: float = 2.0,
) -> list[dict]:
    """Detect surprisal spikes (significantly above local mean)."""
    import numpy as np

    if len(surprisals) < 5:
        return []

    mean = np.mean(surprisals)
    std = np.std(surprisals)
    if std < 0.01:
        return []

    spikes = []
    for i, (s, text) in enumerate(zip(surprisals, sentences)):
        z = (s - mean) / std
        if z > z_threshold:
            spikes.append({
                "index": i,
                "surprisal": round(s, 3),
                "z_score": round(float(z), 2),
                "text": text[:120],
            })

    return sorted(spikes, key=lambda x: -x["surprisal"])[:10]


def analyze_chapter(
    text: str,
    filename: str,
    model_manager,
    mean_words_per_sentence: int = 18,
) -> InfoContourResult:
    """Analyze the information contour of a chapter.

    Args:
        text: chapter text
        filename: for reporting
        model_manager: provides GPT-2
        mean_words_per_sentence: for estimating words-per-cycle
    """
    import numpy as np

    model, tokenizer = model_manager.gpt2
    device = model_manager.device

    items = split_paragraphs_with_breaks(text)
    nlp = model_manager.spacy

    # Extract sentences
    sentences: list[str] = []
    for item in items:
        if item is None:
            continue
        doc = nlp(item)
        for sent in doc.sents:
            st = sent.text.strip()
            if len(st.split()) >= 4:
                sentences.append(st)

    if len(sentences) < 10:
        return InfoContourResult(
            filename=filename,
            sentence_count=len(sentences),
            sentence_surprisals=[],
            mean_surprisal=0.0,
            std_surprisal=0.0,
            cv_surprisal=0.0,
            dominant_period=0,
            dominant_period_words=0,
            rhythmicity=0.0,
            spectral_entropy=1.0,
        )

    # Score each sentence
    print(f"  Scoring {len(sentences)} sentences...", file=sys.stderr, flush=True)
    surprisals = []
    for sent in sentences:
        s = _sentence_surprisal(sent, model, tokenizer, device)
        surprisals.append(s)

    mean_s = float(np.mean(surprisals))
    std_s = float(np.std(surprisals))
    cv_s = std_s / mean_s if mean_s > 0 else 0.0

    # Spectral analysis
    dominant_period, rhythmicity, spectral_entropy = _analyze_spectrum(surprisals)
    dominant_words = dominant_period * mean_words_per_sentence

    # Detect flatlines and spikes
    flatlines = _detect_flatlines(surprisals)
    spikes = _detect_spikes(surprisals, sentences)

    return InfoContourResult(
        filename=filename,
        sentence_count=len(sentences),
        sentence_surprisals=[round(s, 3) for s in surprisals],
        mean_surprisal=round(mean_s, 3),
        std_surprisal=round(std_s, 3),
        cv_surprisal=round(cv_s, 3),
        dominant_period=dominant_period,
        dominant_period_words=dominant_words,
        rhythmicity=round(rhythmicity, 3),
        spectral_entropy=round(spectral_entropy, 3),
        flatlines=flatlines,
        spikes=spikes,
    )
