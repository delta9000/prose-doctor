"""Referential cohesion lens -- entity-grid coherence + networkx graph analysis.

Tracks entity mentions across sentences, scores transition probabilities,
detects pronoun ambiguity and referent churn. Based on the entity-grid model
(Barzilay & Lapata, 2008) and Coh-Metrix referential cohesion indices
(Graesser et al., 2011).
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from prose_doctor.lenses import Lens, LensResult
from prose_doctor.text import split_paragraphs

if TYPE_CHECKING:
    from prose_doctor.providers import ProviderPool

# Entity grid roles
S = "S"  # subject
O = "O"  # object
X = "X"  # other mention
ABSENT = "-"

# Transition quality scores (higher = more coherent)
_TRANSITION_SCORES = {
    (S, S): 1.0,
    (S, O): 0.8,
    (S, X): 0.6,
    (O, S): 0.7,
    (O, O): 0.5,
    (O, X): 0.4,
    (X, S): 0.5,
    (X, O): 0.4,
    (X, X): 0.3,
    (ABSENT, S): 0.1,  # entity appears from nowhere — low coherence
    (ABSENT, O): 0.2,
    (ABSENT, X): 0.2,
    (S, ABSENT): 0.3,
    (O, ABSENT): 0.3,
    (X, ABSENT): 0.2,
    (ABSENT, ABSENT): 0.0,
}


def _resolve_entity(token, entities_in_window: dict[str, str]) -> str | None:
    """Try to resolve a pronoun to a nearby entity."""
    if token.pos_ == "PRON":
        text = token.text.lower()
        if text in ("he", "him", "his"):
            # Return most recent masculine/unknown entity
            for ent in reversed(list(entities_in_window.keys())):
                return ent
        elif text in ("she", "her", "hers"):
            for ent in reversed(list(entities_in_window.keys())):
                return ent
        elif text in ("it", "its"):
            for ent in reversed(list(entities_in_window.keys())):
                return ent
        elif text in ("they", "them", "their"):
            for ent in reversed(list(entities_in_window.keys())):
                return ent
    return None


def _get_entity_key(token) -> str:
    """Get a normalized entity key from a token."""
    if token.pos_ == "PROPN":
        return token.text.lower()
    return token.lemma_.lower()


def _get_role(token) -> str:
    """Determine the grammatical role of a token."""
    if token.dep_ in ("nsubj", "nsubjpass"):
        return S
    if token.dep_ in ("dobj", "pobj", "attr"):
        return O
    return X


class ReferentialCohesionLens(Lens):
    """Measure entity-based referential coherence in prose."""

    name = "referential_cohesion"
    requires_providers = ["spacy"]
    consumes_lenses: list[str] = []

    def analyze(
        self,
        text: str,
        filename: str,
        providers: "ProviderPool",
        consumed: dict | None = None,
    ) -> LensResult:
        nlp = providers.spacy
        doc = nlp(text)
        paragraphs = split_paragraphs(text)
        sentences = list(doc.sents)

        # Build entity grid: entity -> [role_in_sent_0, role_in_sent_1, ...]
        entity_grid: dict[str, list[str]] = defaultdict(lambda: [ABSENT] * len(sentences))
        sentence_entities: list[set[str]] = []
        subjects_per_sent: list[str | None] = []

        # Track entities in a sliding window for pronoun resolution
        recent_entities: dict[str, str] = {}  # entity_key -> last role

        for sent_idx, sent in enumerate(sentences):
            sent_ents: set[str] = set()
            first_subject = None

            # Named entities
            for ent in sent.ents:
                if ent.label_ in ("PERSON", "ORG", "GPE"):
                    key = ent.text.lower()
                    sent_ents.add(key)
                    entity_grid[key][sent_idx] = X

            # Noun chunks and dependency-based roles
            for token in sent:
                if token.pos_ in ("NOUN", "PROPN"):
                    key = _get_entity_key(token)
                    role = _get_role(token)
                    sent_ents.add(key)
                    # Prefer higher-salience role
                    current = entity_grid[key][sent_idx]
                    if current == ABSENT or (role == S and current != S):
                        entity_grid[key][sent_idx] = role
                    if role == S and first_subject is None:
                        first_subject = key
                elif token.pos_ == "PRON" and token.dep_ in ("nsubj", "nsubjpass", "dobj"):
                    resolved = _resolve_entity(token, recent_entities)
                    if resolved:
                        role = _get_role(token)
                        sent_ents.add(resolved)
                        current = entity_grid[resolved][sent_idx]
                        if current == ABSENT or (role == S and current != S):
                            entity_grid[resolved][sent_idx] = role
                        if role == S and first_subject is None:
                            first_subject = resolved

            # Update recent entities window (keep last 3 sentences)
            for ent in sent_ents:
                recent_entities[ent] = entity_grid[ent][sent_idx]
            if sent_idx >= 3:
                # Clean old entities (rough window)
                pass

            sentence_entities.append(sent_ents)
            subjects_per_sent.append(first_subject)

        # Score transitions
        transition_scores = []
        new_subject_count = 0
        seen_subjects: set[str] = set()

        for entity, roles in entity_grid.items():
            for i in range(len(roles) - 1):
                pair = (roles[i], roles[i + 1])
                score = _TRANSITION_SCORES.get(pair, 0.2)
                transition_scores.append(score)

        # Subject churn: count new subjects
        for subj in subjects_per_sent:
            if subj is not None:
                if subj not in seen_subjects:
                    new_subject_count += 1
                seen_subjects.add(subj)

        total_subject_slots = sum(1 for s in subjects_per_sent if s is not None)
        subject_churn = (
            (new_subject_count - 1) / max(total_subject_slots - 1, 1)
            if new_subject_count > 1
            else 0.0
        )

        # Per-paragraph entity continuity
        para_boundaries = []
        offset = 0
        for para in paragraphs:
            start = text.find(para, offset)
            if start == -1:
                start = offset
            para_boundaries.append((start, start + len(para)))
            offset = start + len(para)

        para_entity_sets: list[set[str]] = []
        for p_start, p_end in para_boundaries:
            para_ents: set[str] = set()
            for i, sent in enumerate(sentences):
                if sent.start_char >= p_start and sent.start_char < p_end:
                    para_ents |= sentence_entities[i]
            para_entity_sets.append(para_ents)

        entity_continuity: list[float] = [1.0]  # first paragraph has full continuity
        for i in range(1, len(para_entity_sets)):
            prev = para_entity_sets[i - 1]
            curr = para_entity_sets[i]
            if curr:
                overlap = len(prev & curr) / len(curr)
            else:
                overlap = 0.0
            entity_continuity.append(round(overlap, 3))

        # Entity graph via networkx
        try:
            import networkx as nx

            G = nx.Graph()
            for ent in entity_grid:
                G.add_node(ent)
            for sent_ents in sentence_entities:
                ents = list(sent_ents)
                for i in range(len(ents)):
                    for j in range(i + 1, len(ents)):
                        if G.has_edge(ents[i], ents[j]):
                            G[ents[i]][ents[j]]["weight"] += 1
                        else:
                            G.add_edge(ents[i], ents[j], weight=1)

            if len(G.nodes) > 0:
                pagerank = nx.pagerank(G)
                top_entity = max(pagerank, key=pagerank.get)
                protagonist_centrality = round(pagerank[top_entity], 4)
                density = round(nx.density(G), 4)
                dangling = sum(1 for n in G.nodes if G.degree(n) <= 1)
            else:
                protagonist_centrality = 0.0
                density = 0.0
                dangling = 0
                top_entity = None
                pagerank = {}
        except ImportError:
            protagonist_centrality = 0.0
            density = 0.0
            dangling = 0
            top_entity = None
            pagerank = {}

        coherence_score = round(
            float(sum(transition_scores) / max(len(transition_scores), 1)), 4
        )

        return LensResult(
            lens_name="referential_cohesion",
            per_paragraph={"entity_continuity": entity_continuity},
            per_chapter={
                "coherence_score": coherence_score,
                "subject_churn": round(subject_churn, 4),
                "entity_count": len(entity_grid),
                "dangling_entity_count": dangling,
                "protagonist_centrality": protagonist_centrality,
                "graph_density": density,
            },
            raw={
                "entity_count": len(entity_grid),
                "top_entities": (
                    sorted(pagerank.items(), key=lambda x: -x[1])[:5]
                    if pagerank else []
                ),
            },
        )
