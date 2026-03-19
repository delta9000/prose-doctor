# prose-doctor

Prose analysis toolkit for fiction writers using LLMs. Catches the patterns that make AI-generated prose feel samey: rhetorical tics, vocabulary crutches, density problems, and structural repetition.

Two tiers:
- **Core** (zero dependencies): regex pattern detection, vocabulary crutch finder, density budgets
- **ML** (torch/transformers/spacy): 8-class slop classifier, GPT-2 perplexity scoring, foregrounding index, twin-finder, voice separation, emotion arcs

## Install

```bash
pip install prose-doctor        # core only
pip install prose-doctor[ml]    # everything
```

## Quick start

```bash
# Scan chapters for prose patterns (no ML needed)
prose-doctor scan ./chapters/

# Full analysis with ML models
prose-doctor scan --deep ./chapters/

# JSON output for piping
prose-doctor scan --json ./chapters/ | jq '.summary'

# Generate a config file
prose-doctor init
```

## Commands

| Command | Tier | What it does |
|---------|------|--------------|
| `scan` | core | Pattern detection + vocabulary crutches + density budgets |
| `scan --deep` | ml | Adds classifier, foregrounding, perplexity, emotion arc |
| `index` | ml | Foregrounding index: literary texture across 5 axes |
| `twins` | ml | Find your own best writing on the same topic as a flat passage |
| `classify` | ml | Run the 8-class slop classifier on paragraphs |
| `init` | core | Generate a `.prose-doctor.toml` config template |

## What it catches

### Patterns (30+ rules)

Rhetorical negation ("Not a sound. A structure."), thesis-statement constructions ("It wasn't X. It was Y."), emotion naming ("felt a surge of dread"), phantom sensations ("The truth settled in his chest"), narrator glossing ("Something in her shifted"), weak verbs ("began to", "seemed to"), adverb dialogue tags ("said quietly"), and more.

### Vocabulary crutches

Any word used more than N times per 1000 words, with configurable exemptions for character names and common English.

### Density budgets

Patterns that are fine once but become tics through repetition: tricolons, over-resolved similes, cost-recovery formulas, prepositional stacking. Each has a per-2000-word budget.

### ML scoring

- **Slop classifier**: [ModernBERT fine-tuned on 8 categories](https://huggingface.co/dt9k/prose-doctor-slop-classifier) of LLM prose failure. Context-aware (sees surrounding paragraphs).
- **Perplexity**: GPT-2 flags paragraphs that are suspiciously predictable.
- **Foregrounding index**: Measures literary texture across alliteration, inversion, rhythm variety, fragments, and unexpected collocations.
- **Twin-finder**: For each flat passage, finds a topically similar passage from elsewhere in your manuscript that scores higher on texture. Your own best writing as the rewrite target.
- **Voice separation**: Measures whether POV characters sound distinct via sentence embeddings.
- **Emotion arc**: Flags chapters with flat emotional arcs.

## Configuration

Create a `.prose-doctor.toml` in your project root (or run `prose-doctor init`):

```toml
[prose-doctor]
character_names = ["Rook", "Cassian", "Denna"]
exempt_words = ["rook", "cassian", "denna"]

[prose-doctor.pov]
Rook = ["chapter_01", "chapter_04"]
Cassian = ["chapter_02", "chapter_06"]

[prose-doctor.density_budgets]
tricolon = 2

[prose-doctor.models]
slop_classifier = "dt9k/prose-doctor-slop-classifier"
```

Character names are injected into patterns (e.g., "something in [Rook]") and exempted from vocabulary crutch detection.

## As a library

```python
from prose_doctor.analyzers.doctor import diagnose
from prose_doctor.config import ProjectConfig

config = ProjectConfig(character_names=["Rook", "Cassian"])
health = diagnose(open("chapter.md").read(), filename="chapter.md", config=config)

print(health.total_issues)
print(health.vocabulary_crutches)
print(health.pattern_hits)
```

## License

MIT (the [slop classifier model](https://huggingface.co/dt9k/prose-doctor-slop-classifier) is Apache-2.0)
