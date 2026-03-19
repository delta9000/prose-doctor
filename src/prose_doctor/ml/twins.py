"""Vector similarity twin-finder: find textured twins for flat passages."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from prose_doctor.text import split_paragraphs


@dataclass
class TwinMatch:
    """A matched pair: flat passage and its textured twin."""

    flat_file: str
    flat_idx: int
    flat_text: str
    flat_texture: float
    twin_file: str
    twin_idx: int
    twin_text: str
    twin_texture: float
    topical_similarity: float


def _quick_texture(text: str, nlp) -> float:
    """Fast texture proxy: sentence length CV + opener diversity."""
    import numpy as np

    doc = nlp(text)
    sents = [s for s in doc.sents if len(s) > 2]
    if len(sents) < 2:
        return 0.5
    lengths = [len(s) for s in sents]
    cv = float(np.std(lengths) / max(np.mean(lengths), 1))
    openers = [s[0].pos_ for s in sents]
    div = len(set(openers)) / max(len(openers), 1)
    return cv * 0.6 + div * 0.4


def find_twins(
    files: list[Path],
    model_manager,
    max_results: int = 20,
) -> list[TwinMatch]:
    """Find textured twins for flat paragraphs across all files."""
    import numpy as np
    from numpy.linalg import norm

    nlp = model_manager.spacy
    st_model = model_manager.sentence_transformer

    # Collect all paragraphs
    all_paras: list[tuple[str, int, str]] = []
    for f in files:
        for pi, p in enumerate(split_paragraphs(f.read_text())):
            if len(p.split()) > 12:
                all_paras.append((f.name, pi, p))

    if len(all_paras) < 50:
        return []

    # Embed
    texts = [p[2][:300] for p in all_paras]
    print(f"  Embedding {len(texts)} paragraphs...", file=sys.stderr, flush=True)
    embeddings = st_model.encode(texts, show_progress_bar=False, batch_size=64)

    # Score texture
    print("  Scoring texture...", file=sys.stderr, flush=True)
    scores = np.array([_quick_texture(p[2], nlp) for p in all_paras])

    hi_thresh = np.percentile(scores, 85)
    lo_thresh = np.percentile(scores, 15)

    hi_mask = scores >= hi_thresh
    lo_mask = scores <= lo_thresh

    if hi_mask.sum() < 5 or lo_mask.sum() < 5:
        return []

    centroid_hi = np.mean(embeddings[hi_mask], axis=0)
    centroid_lo = np.mean(embeddings[lo_mask], axis=0)
    quality_dir = centroid_hi - centroid_lo
    quality_dir /= norm(quality_dir)

    # Find twins for flat paragraphs
    flat_indices = [
        (float(np.dot(embeddings[i], quality_dir)), i)
        for i in range(len(all_paras))
        if scores[i] < lo_thresh and len(all_paras[i][2].split()) > 25
    ]
    flat_indices.sort()

    results = []
    for _, wi in flat_indices:
        if len(results) >= max_results:
            break

        fn, pi, wt = all_paras[wi]
        we = embeddings[wi]

        best_sim, best_j = -1.0, -1
        for j in range(len(all_paras)):
            if j == wi or scores[j] < hi_thresh:
                continue
            if all_paras[j][0] == fn and abs(all_paras[j][1] - pi) < 5:
                continue
            s = float(np.dot(we, embeddings[j]) / (norm(we) * norm(embeddings[j])))
            if s > best_sim:
                best_sim, best_j = s, j

        if best_j >= 0 and best_sim > 0.35:
            tn, tpi, tt = all_paras[best_j]
            results.append(TwinMatch(
                flat_file=fn, flat_idx=pi, flat_text=wt,
                flat_texture=float(scores[wi]),
                twin_file=tn, twin_idx=tpi, twin_text=tt,
                twin_texture=float(scores[best_j]),
                topical_similarity=best_sim,
            ))

    return results
