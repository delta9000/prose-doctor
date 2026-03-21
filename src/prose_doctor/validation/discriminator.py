"""Statistical discrimination between human and LLM prose."""
from __future__ import annotations
import numpy as np
from scipy import stats as scipy_stats


def compute_discrimination(human_scores: list[float], llm_scores: list[float]) -> dict:
    h = np.array(human_scores)
    l = np.array(llm_scores)
    pooled_std = np.sqrt((h.std()**2 + l.std()**2) / 2)
    d = (h.mean() - l.mean()) / pooled_std if pooled_std > 0 else 0.0
    u_stat, p_value = scipy_stats.mannwhitneyu(h, l, alternative="two-sided")
    ks_stat, ks_p = scipy_stats.ks_2samp(h, l)
    return {
        "cohens_d": round(float(d), 3),
        "p_value": round(float(p_value), 6),
        "u_statistic": float(u_stat),
        "ks_statistic": round(float(ks_stat), 3),
        "ks_p_value": round(float(ks_p), 6),
        "human_mean": round(float(h.mean()), 4),
        "llm_mean": round(float(l.mean()), 4),
        "human_n": len(h),
        "llm_n": len(l),
    }
