"""Shared fixtures for prose-doctor tests."""

import pytest


SAMPLE_CHAPTER = """\
---
title: Chapter 1: The Last Expedition
book: Book 1
POV: Rook
---

The bunker smelled of copper and old concrete. Rook pressed his palm against the wall, feeling the steady hum that had been there since before the Cataclysm.

Not a sound. A structure. Something that had settled into the bones of the building itself.

---

"We should keep moving," Cassian said quietly. His hand went to the weapon at his hip, a reflex born of years in the Wastes.

Rook didn't answer. He felt the weight of the observation settle behind his ribs, a cartographer's habit of noticing what others missed. The truth sat in his chest like something that had nothing to do with breathing.

She seemed to understand what he meant. That was what it meant to know someone — not the words, but the weight of the silence between them.

Before, the corridors had been empty. Now they pulsed with a patient hum, deep, resonant and vast.

The glass began to slope downward, through the tunnel, into the chamber, against the far wall. Something in Rook shifted, something like recognition.
"""

SAMPLE_PLAIN = """\
The forest was quiet. Birds had stopped singing hours ago.

She walked through the undergrowth, feeling the branches catch at her sleeves. Every step took her further from the path.

Not a trail. A memory of one. The kind that existed only because someone had walked it once, long ago.

The light began to fade.
"""


@pytest.fixture
def sample_chapter():
    return SAMPLE_CHAPTER


@pytest.fixture
def sample_plain():
    return SAMPLE_PLAIN
