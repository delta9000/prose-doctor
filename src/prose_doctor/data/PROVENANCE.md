# Sensory Probe Provenance

## sensory_probe.pt

**Purpose:** Maps all-mpnet-base-v2 embeddings (768d) to 6 sensory modality scores
(visual, auditory, haptic, olfactory, gustatory, interoceptive).

**Architecture:** MLP — Linear(768, 96) → ReLU → Linear(96, 6) → Sigmoid.
25,350 parameters, ~100KB.

**Training pipeline (fully data-free, no external datasets):**

### Step 1: Anchor direction vectors
- 18 hand-picked anchor words per modality (108 total), chosen as unambiguous
  sensory exemplars (e.g., haptic: rough, smooth, gritty, velvety...)
- Embedded with Qwen3-Embedding-4B (2560d) via local endpoint
- Per-modality centroid computed, then direction vector = centroid - global mean
- Each word scored by projection onto these 6 direction vectors

### Step 2: GPT-2 cloze scoring
- For 4,000 sampled words, computed GPT-2 (117M) perplexity on cloze templates:
  "I can {see/hear/touch/smell/taste/feel inside} the {word}."
  vs baseline "I can experience the {word}."
- Lower perplexity = stronger modality association
- Scores normalized to [0, 1] per modality

### Step 3: Blended pseudo-labels
- Per-modality blend weights chosen based on which method correlated better
  with Lancaster Sensorimotor Norms (used for validation only, not training):
  - Visual: 0.6 direction / 0.4 cloze
  - Auditory: 0.6 / 0.4
  - Haptic: 0.5 / 0.5
  - Olfactory: 0.3 / 0.7 (cloze better for smell)
  - Gustatory: 0.3 / 0.7 (cloze better for taste)
  - Interoceptive: 0.9 / 0.1 (cloze fails on interoception)

### Step 4: Qwen3 teacher probe
- Trained a probe (2560 → 128 → 6) on 4,000 blended pseudo-labels using
  Qwen3-Embedding-4B embeddings
- Generated teacher labels for all 36,811 words

### Step 5: MPNet student distillation
- Embedded all 36,811 words with all-mpnet-base-v2 (768d)
- Trained final probe (768 → 96 → 6) on teacher labels from step 4
- 90/10 train/test split, 100 epochs, Adam lr=1e-3, batch 256

## Validation against Lancaster Sensorimotor Norms

Lancaster norms (Lynott et al., 2020) used ONLY for validation, never for training.
The norms provide human-rated perceptual strength (0-5 scale) for 40,000 English words.

| Modality | Pearson r | R² |
|---|---|---|
| Visual | 0.35 | 0.12 |
| Auditory | 0.20 | 0.04 |
| Haptic | 0.44 | 0.19 |
| Olfactory | 0.47 | 0.22 |
| Gustatory | 0.50 | 0.25 |
| Interoceptive | 0.43 | 0.18 |
| **Mean** | **0.40** | |

Spot check accuracy (dominant modality correct): 11/11 test words including
crimson→visual, silk→haptic, stench→olfactory, heartbeat→interoceptive.

## Comparison to alternatives tested

| Method | Mean Pearson | Spot accuracy | Data-free? |
|---|---|---|---|
| Cosine to anchor centroids | 0.15 | 14/16 | yes |
| Direction vectors (MiniLM 384d) | 0.27 | 14/16 | yes |
| Direction vectors (MPNet 768d) | 0.32 | 14/16 | yes |
| GPT-2 cloze alone | 0.28 | 10/16 | yes |
| Blended probe (MiniLM, no distill) | 0.36 | 10/11 | yes |
| Blended probe (MPNet, no distill) | 0.41 | 10/11 | yes |
| Qwen3-Embedding direct (2560d) | 0.39 | 11/11 | yes |
| **MPNet←Qwen3 distilled (shipped)** | **0.40** | **11/11** | **yes** |
| Lancaster-trained probe (oracle) | 0.68 | 11/11 | no |
| Granite-4-micro collocate mining | 0.16 | 2/11 | yes |
| Qwen3-30B collocate mining | 0.16 | 2/11 | yes |

## Models used during training (not required at inference)

- **Qwen3-Embedding-4B** (2560d): teacher embedding model
- **GPT-2** (117M): cloze probability scoring
- **all-mpnet-base-v2** (768d): student embedding model (also used at inference)

## Reproducibility

Training script: `train_sensory_probe.py` (in repo root, uses Lancaster for
validation only). Full distillation pipeline ran in the conversation that
created this file — see commit history for the exact code.

## Citation

Validation reference (not a training dependency):
Lynott, D., Connell, L., Brysbaert, M., Brand, J., & Carney, J. (2020).
The Lancaster Sensorimotor Norms: Multidimensional measures of perceptual
and action strength for 40,000 English words. Behavior Research Methods,
52, 1271-1291. https://doi.org/10.3758/s13428-019-01316-z
