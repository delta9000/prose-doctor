"""Batch attention fingerprinting: run on multiple files, output comparison table.

Usage:
    uv run python batch_attention.py file1.md file2.md ...
    uv run python batch_attention.py corpus/baselines/watts*.md  # glob
"""
from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

from narrative_attention_proto import build_paragraph_features, compute_attention
from compare_attention import attention_fingerprint


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python batch_attention.py file1.md file2.md ...")
        sys.exit(1)

    files = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.glob("*.md")))
        else:
            import glob
            files.extend(Path(f) for f in sorted(glob.glob(arg)))

    if not files:
        print("No files found.")
        sys.exit(1)

    results = []
    for f in files:
        text = f.read_text()
        # Skip files that are too short
        from prose_doctor.text import split_paragraphs
        paras = split_paragraphs(text)
        if len(paras) < 10:
            print(f"  Skipping {f.name} ({len(paras)} paragraphs, too short)", file=sys.stderr)
            continue

        print(f"  Processing {f.name} ({len(paras)} paragraphs)...", file=sys.stderr, flush=True)
        try:
            data = build_paragraph_features(text, f.name)
            attn = compute_attention(data["structural"])
            fp = attention_fingerprint(attn, data["raw"], data["scene_breaks"])
            fp["filename"] = f.name
            results.append(fp)
        except Exception as e:
            print(f"  ERROR on {f.name}: {e}", file=sys.stderr)

    if not results:
        print("No valid results.")
        sys.exit(1)

    # Print comparison table
    keys = [
        ("File", "filename", False),
        ("Paras", "n_paragraphs", False),
        ("Scenes", "n_scenes", False),
        ("BlkDiag", "block_diagonal", True),
        ("Temp", "temperature", True),
        ("Variety", "structural_variety", True),
        ("Orphans", "n_orphans", False),
        ("Generics", "n_generics", False),
        ("AttnStd", "attn_std", True),
        ("RowEntMu", "row_entropy_mean", True),
        ("RowEntSd", "row_entropy_std", True),
    ]

    # Header
    header = ""
    for label, _, _ in keys:
        if label == "File":
            header += f"{'File':<45}"
        else:
            header += f"{label:>9}"
    print(header)
    print("-" * len(header))

    for r in results:
        line = ""
        for label, key, is_float in keys:
            val = r.get(key, "")
            if label == "File":
                line += f"{str(val):<45}"
            elif is_float:
                line += f"{val:>9.4f}"
            else:
                line += f"{val:>9}"

        print(line)

    # Summary statistics by "source" if we can infer it
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")

    all_temps = [r["temperature"] for r in results]
    all_variety = [r["structural_variety"] for r in results]
    all_bd = [r["block_diagonal"] for r in results]
    all_entropy = [r["row_entropy_mean"] for r in results]
    all_generics = [r["n_generics"] / r["n_paragraphs"] for r in results]

    print(f"  Temperature:    mean={np.mean(all_temps):.3f} std={np.std(all_temps):.3f} range=[{min(all_temps):.3f}, {max(all_temps):.3f}]")
    print(f"  Variety:        mean={np.mean(all_variety):.4f} std={np.std(all_variety):.4f}")
    print(f"  Block diagonal: mean={np.mean(all_bd):.4f} std={np.std(all_bd):.4f}")
    print(f"  Row entropy:    mean={np.mean(all_entropy):.3f} std={np.std(all_entropy):.3f}")
    print(f"  Generic ratio:  mean={np.mean(all_generics):.1%} std={np.std(all_generics):.1%}")


if __name__ == "__main__":
    main()
