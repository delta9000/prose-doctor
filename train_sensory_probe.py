#!/usr/bin/env python3
"""Train a sensory modality probe on Lancaster norms.

Takes sentence-transformer embeddings of 40k words, learns a projection
to 6 perceptual dimensions (visual, auditory, haptic, olfactory,
gustatory, interoceptive).

Usage:
    uv run --with prose-doctor[ml] python train_sensory_probe.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

NORMS_PATH = Path("refs/Lancaster_sensorimotor_norms_for_39707_words.csv")
OUTPUT_DIR = Path("src/prose_doctor/data")
MODALITIES = ["Auditory", "Gustatory", "Haptic", "Interoceptive", "Olfactory", "Visual"]
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class SensoryProbe(nn.Module):
    """Small MLP: embedding → 6 sensory scores."""

    def __init__(self, input_dim: int = EMBEDDING_DIM, hidden_dim: int = 64, output_dim: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),  # output 0-1
        )

    def forward(self, x):
        return self.net(x)


def load_norms() -> tuple[list[str], np.ndarray]:
    """Load Lancaster norms, return (words, scores[N, 6])."""
    words = []
    scores = []

    with open(NORMS_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["Word"].strip()
            # Skip multi-word entries
            if " " in word:
                continue
            try:
                vals = [float(row[f"{m}.mean"]) / 5.0 for m in MODALITIES]
            except (ValueError, KeyError):
                continue
            words.append(word.lower())
            scores.append(vals)

    return words, np.array(scores, dtype=np.float32)


def main():
    print("Loading Lancaster norms...")
    words, scores = load_norms()
    print(f"  {len(words)} single words loaded")
    print(f"  Score ranges: {scores.min(axis=0).round(3)} to {scores.max(axis=0).round(3)}")

    print("\nEmbedding words with all-MiniLM-L6-v2...")
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = st.encode(words, show_progress_bar=True, batch_size=256)
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Train/test split
    n = len(words)
    indices = np.random.RandomState(42).permutation(n)
    split = int(n * 0.9)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = torch.from_numpy(embeddings[train_idx])
    y_train = torch.from_numpy(scores[train_idx])
    X_test = torch.from_numpy(embeddings[test_idx])
    y_test = torch.from_numpy(scores[test_idx])

    print(f"\n  Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Train
    model = SensoryProbe()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

    print("\nTraining...")
    for epoch in range(50):
        model.train()
        losses = []
        for xb, yb in train_dl:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(X_test)
                test_loss = criterion(test_pred, y_test).item()
                # Per-modality R²
                y_np = y_test.numpy()
                p_np = test_pred.numpy()
                r2s = []
                for i, m in enumerate(MODALITIES):
                    ss_res = ((y_np[:, i] - p_np[:, i]) ** 2).sum()
                    ss_tot = ((y_np[:, i] - y_np[:, i].mean()) ** 2).sum()
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    r2s.append(r2)
                r2_str = " ".join(f"{m[:3]}={r2:.3f}" for m, r2 in zip(MODALITIES, r2s))
            print(f"  Epoch {epoch+1:>3}: train_loss={np.mean(losses):.5f} "
                  f"test_loss={test_loss:.5f} R²: {r2_str}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        y_np = y_test.numpy()
        p_np = test_pred.numpy()

    print(f"\nFinal per-modality results:")
    print(f"  {'Modality':<15} {'R²':>8} {'MAE':>8} {'Corr':>8}")
    for i, m in enumerate(MODALITIES):
        ss_res = ((y_np[:, i] - p_np[:, i]) ** 2).sum()
        ss_tot = ((y_np[:, i] - y_np[:, i].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        mae = np.abs(y_np[:, i] - p_np[:, i]).mean()
        corr = np.corrcoef(y_np[:, i], p_np[:, i])[0, 1]
        print(f"  {m:<15} {r2:>8.4f} {mae:>8.4f} {corr:>8.4f}")

    # Spot-check some words
    print("\nSpot checks:")
    check_words = ["crimson", "whisper", "rough", "cinnamon", "nausea", "bright",
                    "abstract", "democracy", "thunder", "silk", "stench", "heartbeat"]
    check_embs = st.encode(check_words)
    with torch.no_grad():
        check_pred = model(torch.from_numpy(np.array(check_embs, dtype=np.float32)))

    header = f"  {'Word':<15}" + "".join(f" {m[:4]:>6}" for m in MODALITIES)
    print(header)
    for word, pred in zip(check_words, check_pred.numpy()):
        vals = "".join(f" {v:>6.3f}" for v in pred)
        dominant = MODALITIES[pred.argmax()][:4]
        print(f"  {word:<15}{vals}  ← {dominant}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "sensory_probe.pt"
    torch.save(model.state_dict(), model_path)

    meta = {
        "modalities": MODALITIES,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": 64,
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "source": "Lancaster Sensorimotor Norms (Lynott et al. 2020)",
    }
    with open(OUTPUT_DIR / "sensory_probe_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    param_count = sum(p.numel() for p in model.parameters())
    file_size = model_path.stat().st_size
    print(f"\nSaved to {model_path}")
    print(f"  Parameters: {param_count:,} ({file_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
