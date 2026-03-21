"""Compute the concreteness direction vector in mpnet-base-v2 embedding space.

Uses anchor words to find the abstract→concrete axis, then validates
against Brysbaert norms. Saves the unit direction vector as .npy.

Usage:
    cd /home/ben/code/prose-doctor
    uv run python scripts/compute_concreteness_direction.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer

# Anchor words — chosen to span clear concrete/abstract exemplars
# avoiding genre-specific terms or words with strong polysemy
CONCRETE_ANCHORS = [
    "hammer", "blood", "kitchen", "cigarette", "gravel", "elbow",
    "rust", "doorknob", "shovel", "brick", "fingernail", "puddle",
    "collar", "windshield", "sandpaper", "kettle", "splinter", "ankle",
]
ABSTRACT_ANCHORS = [
    "freedom", "justice", "possibility", "tendency", "significance",
    "notion", "truth", "essence", "irony", "ambiguity", "paradox",
    "morality", "hypothesis", "obligation", "sentiment", "intuition",
]

OUT_PATH = Path(__file__).resolve().parent.parent / "src" / "prose_doctor" / "data" / "concreteness_direction.npy"
NORMS_PATH = Path(__file__).resolve().parent.parent / "refs" / "brysbaert_concreteness.csv"


def main():
    print("Loading mpnet-base-v2...")
    st = SentenceTransformer("all-mpnet-base-v2")

    # Compute direction from anchors
    concrete_embs = st.encode(CONCRETE_ANCHORS)
    abstract_embs = st.encode(ABSTRACT_ANCHORS)

    concrete_centroid = concrete_embs.mean(axis=0)
    abstract_centroid = abstract_embs.mean(axis=0)

    direction = concrete_centroid - abstract_centroid
    direction = direction / np.linalg.norm(direction)

    print(f"Direction vector shape: {direction.shape}")
    print(f"Direction norm: {np.linalg.norm(direction):.4f}")

    # Validate against Brysbaert norms
    print(f"\nLoading Brysbaert norms from {NORMS_PATH}...")
    df = pd.read_csv(NORMS_PATH)

    # Find the concreteness column (may be named Conc.M or similar)
    conc_col = None
    for col in df.columns:
        if "conc" in col.lower() and ("m" in col.lower() or "mean" in col.lower()):
            conc_col = col
            break
    if conc_col is None:
        print("WARNING: Could not find concreteness column. Available columns:")
        print(df.columns.tolist())
        return

    # Find the word column
    word_col = None
    for col in df.columns:
        if col.lower() == "word":
            word_col = col
            break
    if word_col is None:
        word_col = df.columns[0]

    print(f"Using columns: word={word_col}, concreteness={conc_col}")

    # Sample for validation (encoding 40K words takes a while)
    sample = df.dropna(subset=[conc_col]).sample(n=min(5000, len(df)), random_state=42)
    words = sample[word_col].tolist()
    human_scores = sample[conc_col].values

    print(f"Encoding {len(words)} words for validation...")
    embeddings = st.encode(words, show_progress_bar=True, batch_size=256)
    predicted = embeddings @ direction

    r, p = pearsonr(predicted, human_scores)
    print(f"\nValidation: r={r:.4f}, p={p:.2e}")
    print(f"  (r > 0.70 is good, r > 0.80 is excellent)")

    if r < 0.50:
        print("WARNING: Correlation is low. Direction vector may be unreliable.")
        print("Consider adjusting anchor words or using the Brysbaert lookup directly.")

    # Save
    np.save(OUT_PATH, direction)
    print(f"\nSaved direction vector to {OUT_PATH}")


if __name__ == "__main__":
    main()
