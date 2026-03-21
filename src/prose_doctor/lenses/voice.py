"""POV voice separation analysis using sentence embeddings."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

from prose_doctor.config import ProjectConfig
from prose_doctor.text import split_paragraphs

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool


class VoiceAnalyzer:
    """Measures POV voice separation and tracks per-chapter drift."""

    SIMILARITY_THRESHOLD = 0.90

    def __init__(self, providers: ProviderPool | None = None, config: ProjectConfig | None = None):
        self._providers = providers
        self._config = config or ProjectConfig()

    def _get_pov(self, filename: str) -> str | None:
        """Determine POV character from filename using config."""
        for pov, prefixes in self._config.pov_map.items():
            for prefix in prefixes:
                if prefix in filename:
                    return pov
        return None

    def analyze_voices(self, files: list[Path]) -> dict:
        """Full voice analysis across all chapter files."""
        import numpy as np
        from numpy.linalg import norm

        st_model = self._providers.sentence_transformer if self._providers else None
        if st_model is None:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer("all-MiniLM-L6-v2")

        pov_paras: dict[str, list[tuple[str, str]]] = {}
        chapter_paras: dict[str, list[tuple[str, str]]] = {}

        for f in files:
            pov = self._get_pov(f.name)
            if not pov:
                continue
            paras = split_paragraphs(f.read_text())
            narrative = [
                p[:200] for p in paras
                if not p.startswith('"') and not p.startswith("\u201c")
                and len(p.split()) > 15
            ]
            pov_paras.setdefault(pov, [])
            for p in narrative:
                pov_paras[pov].append((f.name, p))
            chapter_paras[f.name] = [(pov, p) for p in narrative]

        if len(pov_paras) < 2:
            return {"global_similarity": {}, "classification_accuracy": 0, "per_chapter_drift": []}

        # Sample for global analysis
        random.seed(42)
        sampled = {}
        for pov, items in pov_paras.items():
            sampled[pov] = random.sample(items, min(200, len(items)))

        all_texts = []
        all_labels = []
        for pov, items in sampled.items():
            for _, text in items:
                all_texts.append(text)
                all_labels.append(pov)

        embeddings = st_model.encode(all_texts, show_progress_bar=False, batch_size=64)
        pov_names = list(pov_paras.keys())

        centroids = {}
        for pov in pov_names:
            mask = [i for i, l in enumerate(all_labels) if l == pov]
            if mask:
                centroids[pov] = np.mean(embeddings[mask], axis=0)

        # Global similarity
        global_sims = {}
        for i, p1 in enumerate(pov_names):
            for p2 in pov_names[i + 1:]:
                if p1 in centroids and p2 in centroids:
                    sim = float(
                        np.dot(centroids[p1], centroids[p2])
                        / (norm(centroids[p1]) * norm(centroids[p2]))
                    )
                    key = f"{min(p1, p2)}<->{max(p1, p2)}"
                    global_sims[key] = sim

        # Classification accuracy
        correct = 0
        for i, emb in enumerate(embeddings):
            true = all_labels[i]
            dists = {
                p: float(np.dot(emb, centroids[p]) / (norm(emb) * norm(centroids[p])))
                for p in centroids
            }
            if max(dists, key=dists.get) == true:
                correct += 1
        accuracy = correct / len(all_labels) if all_labels else 0

        # Per-chapter drift
        drift_data = []
        for fname, items in chapter_paras.items():
            if not items:
                continue
            pov = items[0][0]
            texts = [t for _, t in items]
            if len(texts) < 5:
                continue
            ch_embs = st_model.encode(texts, show_progress_bar=False)
            ch_centroid = np.mean(ch_embs, axis=0)

            for other_pov in centroids:
                if other_pov == pov:
                    continue
                cross_sim = float(
                    np.dot(ch_centroid, centroids[other_pov])
                    / (norm(ch_centroid) * norm(centroids[other_pov]))
                )
                own_sim = float(
                    np.dot(ch_centroid, centroids[pov])
                    / (norm(ch_centroid) * norm(centroids[pov]))
                )
                drift_data.append({
                    "filename": fname,
                    "pov": pov,
                    "other_pov": other_pov,
                    "own_similarity": own_sim,
                    "cross_similarity": cross_sim,
                    "drift": cross_sim - own_sim,
                })

        drift_data.sort(key=lambda x: x["cross_similarity"], reverse=True)

        return {
            "global_similarity": global_sims,
            "classification_accuracy": accuracy,
            "per_chapter_drift": drift_data,
        }
