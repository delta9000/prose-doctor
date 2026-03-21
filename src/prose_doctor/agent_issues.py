"""Per-passage issue finder for the revision agent.

Given a metric name and a chapter's text + report, returns specific
passages that are problematic — with paragraph indices, text excerpts,
and surrounding context. This lets the agent target edits precisely
instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from prose_doctor.text import split_paragraphs


@dataclass
class Issue:
    """A specific problematic passage."""
    paragraph_idx: int
    sentence_text: str
    context_before: str  # preceding sentence or paragraph excerpt
    context_after: str   # following sentence or paragraph excerpt
    reason: str          # why this is a problem
    preserve: bool       # True = this is likely intentional craft, don't touch


# --- Fragment classifier helpers ---

# Vague/abstract content patterns — almost always crutches
_VAGUE_FRAGMENTS = {
    "the horizon shimmered", "the air changed", "the world shifted",
    "the ground trembled", "the light faded", "the darkness deepened",
    "the silence stretched", "the sound stopped", "the weight settled",
    "the distance grew", "something changed", "everything changed",
    "nothing moved", "time passed", "time stopped",
}

_BODY_PARTS = frozenset({
    "teeth", "jaw", "throat", "chest", "ribs", "spine", "skull",
    "fingers", "knuckles", "wrist", "palm", "hand", "hands",
    "stomach", "gut", "lungs", "knees", "ankle", "bone", "bones",
    "skin", "muscle", "nerve", "ear", "ears", "eye", "eyes",
    "shoulder", "shoulders", "neck", "elbow", "hip", "tongue",
})


def _has_concrete_detail(text: str) -> bool:
    """Does this fragment contain concrete sensory/physical detail?"""
    words = set(text.lower().rstrip(".!?").split())

    # Body parts = concrete
    if words & _BODY_PARTS:
        return True

    # Named character (capitalized word that isn't sentence-start)
    split = text.split()
    if len(split) >= 2 and any(w[0].isupper() and w.isalpha() for w in split[1:]):
        return True

    # Possessive proper noun at start ("Fen's voice") = specific
    if split and "'" in split[0] and split[0][0].isupper():
        return True

    return False


def _is_vague_fragment(text: str) -> bool:
    """Is this fragment generic/abstract with no concrete detail?"""
    normalized = text.lower().rstrip(".!?").strip()
    if normalized in _VAGUE_FRAGMENTS:
        return True

    # "The [abstract noun] [verb]ed." pattern
    words = normalized.split()
    if len(words) <= 3 and words[0] in ("the", "a", "an", "it"):
        return True

    return False


def find_fragment_issues(text: str, report: dict) -> list[Issue]:
    """Find specific fragment sentences and classify as crutch vs intentional.

    Fragments are sentences with < 5 tokens. Default is CRUTCH — a fragment
    must earn preservation through positive signals:
    - 3+ fragment sequence (montage/list)
    - Anaphoric echo with adjacent fragment
    - Dialogue markers or em-dash continuation
    - Concrete sensory/body detail + structural role (in a pair)

    Fragments are crutches if they:
    - Are vague/abstract ("The horizon shimmered.")
    - Are isolated with no structural role
    - Are in a pair but both lack concrete detail
    """
    from prose_doctor.ml.models import ModelManager
    mm = ModelManager()
    nlp = mm.spacy

    paragraphs = split_paragraphs(text)
    issues: list[Issue] = []

    for pi, para in enumerate(paragraphs):
        doc = nlp(para)
        sents = list(doc.sents)
        sent_lengths = [len(s) for s in sents]

        for si, sent in enumerate(sents):
            if len(sent) >= 5:
                continue

            sent_text = sent.text.strip()
            if not sent_text or len(sent_text) < 3:
                continue

            ctx_before = sents[si - 1].text.strip()[:120] if si > 0 else ""
            ctx_after = sents[si + 1].text.strip()[:120] if si < len(sents) - 1 else ""

            preserve = False
            reason = ""

            # --- Hard preserve: dialogue and em-dash ---
            if any(c in sent_text for c in ('"', '\u201c', '\u201d')):
                issues.append(Issue(pi, sent_text, ctx_before, ctx_after,
                                    "dialogue fragment", True))
                continue

            if '\u2014' in sent_text or '--' in sent_text:
                issues.append(Issue(pi, sent_text, ctx_before, ctx_after,
                                    "em-dash continuation", True))
                continue

            # --- Fragment run analysis ---
            run_start = si
            while run_start > 0 and sent_lengths[run_start - 1] < 5:
                run_start -= 1
            run_end = si
            while run_end < len(sents) - 1 and sent_lengths[run_end + 1] < 5:
                run_end += 1
            run_length = run_end - run_start + 1

            # 3+ sequence: preserve (montage/list)
            if run_length >= 3:
                issues.append(Issue(pi, sent_text, ctx_before, ctx_after,
                                    f"part of {run_length}-fragment sequence (montage/list)", True))
                continue

            # Anaphoric echo: shares first word with adjacent fragment
            if run_length >= 2 and si > 0 and sent_lengths[si - 1] < 5:
                prev_first = sents[si - 1].text.strip().split()[0].lower()
                curr_first = sent_text.split()[0].lower()
                if prev_first == curr_first:
                    issues.append(Issue(pi, sent_text, ctx_before, ctx_after,
                                        f"anaphoric echo ('{curr_first}...')", True))
                    continue

            # --- Content-based classification ---
            is_concrete = _has_concrete_detail(sent_text)
            is_vague = _is_vague_fragment(sent_text)

            if run_length == 2 and is_concrete:
                # Pair with concrete detail — preserve
                reason = "fragment pair with concrete detail"
                preserve = True
            elif is_vague:
                reason = "vague/generic fragment — merge into surrounding sentence"
                preserve = False
            elif is_concrete:
                # Isolated but concrete — flag but note it might be intentional
                reason = "fragment with concrete detail — check if rhythmically earned"
                preserve = False  # still flag it, agent decides
            else:
                reason = "isolated fragment — merge into surrounding sentence"
                preserve = False

            issues.append(Issue(pi, sent_text, ctx_before, ctx_after, reason, preserve))

    return issues


def find_psychic_distance_issues(text: str, report: dict) -> list[Issue]:
    """Find sentences where psychic distance is too external, with causes.

    Instead of just pointing at paragraphs, identifies specific sentences
    and explains WHY they're too distant (no perception verbs, no pronouns,
    expository tone, etc).
    """
    from prose_doctor.ml.models import ModelManager
    from prose_doctor.ml.psychic_distance import (
        PERCEPTION_VERBS, COGNITION_VERBS, PROXIMAL_DEICTICS,
    )
    mm = ModelManager()
    nlp = mm.spacy

    paragraphs = split_paragraphs(text)
    pd = report.get("psychic_distance") or {}
    para_means = pd.get("paragraph_means", [])

    baseline = 0.336
    issues = []

    for pi, mean in enumerate(para_means):
        if pi >= len(paragraphs):
            break
        if mean >= baseline - 0.05:
            continue  # close enough to baseline

        para = paragraphs[pi]
        doc = nlp(para)

        for sent in doc.sents:
            tokens = [t for t in sent if not t.is_space]
            if len(tokens) < 5:
                continue

            words = {t.text.lower() for t in tokens}
            sent_text = sent.text.strip()

            # Diagnose WHY this sentence is distant
            causes = []

            has_perception = bool(words & PERCEPTION_VERBS)
            has_cognition = bool(words & COGNITION_VERBS)
            has_proximal = bool(words & PROXIMAL_DEICTICS)
            has_first_person = bool(words & {"i", "me", "my", "mine", "we", "us", "our"})
            has_third_person = any(
                t.pos_ == "PRON" and t.text.lower() not in
                ("i", "me", "my", "mine", "we", "us", "our", "you", "your", "it", "its")
                for t in tokens
            )

            if not has_perception and not has_cognition:
                causes.append("no perception/cognition verbs")
            if not has_first_person and not has_third_person:
                causes.append("no character pronouns")
            if not has_proximal:
                causes.append("no proximal deictics (this/here/now)")

            # Only flag if there are actual causes (not just slightly below baseline)
            if len(causes) >= 2:
                ctx_before = ""
                ctx_after = ""
                issues.append(Issue(
                    paragraph_idx=pi,
                    sentence_text=sent_text[:150],
                    context_before=ctx_before,
                    context_after=ctx_after,
                    reason=f"too distant — {', '.join(causes)}",
                    preserve=False,
                ))

    # Sort by paragraph (narrative order), limit to worst 15
    issues.sort(key=lambda i: i.paragraph_idx)
    return issues[:15]


def find_inversion_issues(text: str, report: dict) -> list[Issue]:
    """Find specific SVO sentences that could be inverted, with context."""
    from prose_doctor.ml.models import ModelManager
    mm = ModelManager()
    nlp = mm.spacy

    paragraphs = split_paragraphs(text)
    issues = []

    for pi, para in enumerate(paragraphs):
        if len(para.split()) < 20:
            continue

        doc = nlp(para)
        sents = list(doc.sents)
        total = len(sents)
        if total < 3:
            continue

        # Find SVO sentences in this paragraph
        svo_sents = []
        for si, sent in enumerate(sents):
            if len(sent) < 5:
                continue
            root = None
            is_inverted = False
            for t in sent:
                if t.dep_ == "ROOT":
                    root = t
                if t.dep_ in ("nsubj", "nsubjpass") and root is not None and t.i > root.i:
                    is_inverted = True
                    break
            if not is_inverted:
                # Check if it starts with subject
                for t in sent:
                    if t.dep_ in ("nsubj", "nsubjpass"):
                        if t.i <= sent.start + 2:  # subject near the start
                            svo_sents.append((si, sent))
                        break

        inv_count = total - len(svo_sents)
        inv_pct = inv_count / total * 100 if total > 0 else 100

        if inv_pct < 15 and len(svo_sents) >= 3:
            # Pick the best candidates for inversion (longer sentences)
            candidates = sorted(svo_sents, key=lambda x: -len(x[1]))
            for si, sent in candidates[:3]:
                sent_text = sent.text.strip()
                ctx_before = sents[si - 1].text.strip()[:120] if si > 0 else ""
                ctx_after = sents[si + 1].text.strip()[:120] if si < len(sents) - 1 else ""

                # Suggest a specific restructuring
                first_token = sent[0].text if sent else ""
                issues.append(Issue(
                    paragraph_idx=pi,
                    sentence_text=sent_text[:150],
                    context_before=ctx_before,
                    context_after=ctx_after,
                    reason=f"SVO sentence (starts with '{first_token}') — "
                           f"restructure: lead with prepositional phrase, verb, or subordinate clause",
                    preserve=False,
                ))

    return issues[:15]


def find_flatline_issues(text: str, report: dict) -> list[Issue]:
    """Find information flatlines with proper sentence-to-paragraph mapping."""
    from prose_doctor.ml.models import ModelManager
    mm = ModelManager()
    nlp = mm.spacy

    paragraphs = split_paragraphs(text)
    ic = report.get("info_contour") or {}
    flatlines = ic.get("flatlines", [])

    if isinstance(flatlines, int):
        return []

    # Build a sentence→paragraph index map
    sent_to_para: list[int] = []
    for pi, para in enumerate(paragraphs):
        doc = nlp(para)
        for sent in doc.sents:
            if len(sent.text.strip().split()) >= 4:
                sent_to_para.append(pi)

    issues = []
    for fl in flatlines:
        if not isinstance(fl, dict):
            continue
        start = fl.get("start", 0)
        end = fl.get("end", start)

        # Map to paragraph indices properly
        if start < len(sent_to_para):
            para_start = sent_to_para[start]
            para_end = sent_to_para[min(end, len(sent_to_para) - 1)]
            mid_para = (para_start + para_end) // 2

            if mid_para < len(paragraphs):
                para_text = paragraphs[mid_para]
                ctx_before = paragraphs[mid_para - 1][:100] if mid_para > 0 else ""
                ctx_after = paragraphs[mid_para + 1][:100] if mid_para < len(paragraphs) - 1 else ""

                issues.append(Issue(
                    paragraph_idx=mid_para,
                    sentence_text=para_text[:200],
                    context_before=ctx_before,
                    context_after=ctx_after,
                    reason=f"information flatline (sentences {start}-{end}, "
                           f"{fl.get('length', 0)} sentences, "
                           f"mean surprisal {fl.get('mean_surprisal', 0):.3f}) — "
                           f"break with unexpected detail, sensory shift, or register change",
                    preserve=False,
                ))

    return issues


def find_spike_issues(text: str, report: dict) -> list[Issue]:
    """Find paragraphs that are too predictable and need surprisal spikes.

    When a chapter has too few spikes, this identifies the flattest passages
    where injecting surprise would help most.
    """
    from prose_doctor.ml.models import ModelManager
    mm = ModelManager()
    nlp = mm.spacy

    paragraphs = split_paragraphs(text)
    ic = report.get("info_contour") or {}
    surprisals = ic.get("sentence_surprisals", []) if hasattr(ic, 'get') else []

    # If we don't have per-sentence data, fall back to flatlines
    if not surprisals:
        return find_flatline_issues(text, report)

    # Build sentence→paragraph map
    sent_to_para: list[int] = []
    sent_texts: list[str] = []
    for pi, para in enumerate(paragraphs):
        doc = nlp(para)
        for sent in doc.sents:
            st = sent.text.strip()
            if len(st.split()) >= 4:
                sent_to_para.append(pi)
                sent_texts.append(st)

    if not surprisals or len(surprisals) != len(sent_to_para):
        return []

    # Find the lowest-surprisal sentences — these are the most predictable
    import numpy as np
    arr = np.array(surprisals)
    mean_s = float(np.mean(arr))

    # Sentences below mean are candidates for surprise injection
    indexed = [(i, s) for i, s in enumerate(surprisals) if s < mean_s - 0.1]
    indexed.sort(key=lambda x: x[1])  # most predictable first

    issues = []
    seen_paras = set()
    for si, surprisal in indexed[:20]:
        if si >= len(sent_to_para):
            continue
        pi = sent_to_para[si]
        if pi in seen_paras:
            continue
        seen_paras.add(pi)

        sent_text = sent_texts[si] if si < len(sent_texts) else ""
        issues.append(Issue(
            paragraph_idx=pi,
            sentence_text=sent_text[:150],
            context_before="",
            context_after="",
            reason=f"low surprisal ({surprisal:.3f}, mean: {mean_s:.3f}) — "
                   f"predictable phrasing, inject unexpected word choice or syntactic surprise",
            preserve=False,
        ))

    return issues[:10]


METRIC_FINDERS = {
    "fg_fragment": find_fragment_issues,
    "fg_inversion": find_inversion_issues,
    "pd_mean": find_psychic_distance_issues,
    "pd_std": find_psychic_distance_issues,
    "ic_rhythmicity": find_flatline_issues,
    "ic_flatlines": find_flatline_issues,
    "ic_spikes": find_spike_issues,
}


def find_issues(metric: str, text: str, report: dict) -> list[Issue]:
    """Find specific problematic passages for a metric."""
    finder = METRIC_FINDERS.get(metric)
    if finder is None:
        return []
    return finder(text, report)


def format_issues(issues: list[Issue]) -> str:
    """Format issues into a readable string for the agent."""
    if not issues:
        return "No specific issues found for this metric."

    fix = [i for i in issues if not i.preserve]
    preserve = [i for i in issues if i.preserve]

    lines = []

    if fix:
        lines.append(f"## Fix These ({len(fix)} passages)\n")
        for i in fix:
            lines.append(f"**[paragraph {i.paragraph_idx}]** {i.reason}")
            lines.append(f"  > {i.sentence_text}")
            if i.context_before:
                lines.append(f"  BEFORE: {i.context_before}")
            if i.context_after:
                lines.append(f"  AFTER: {i.context_after}")
            lines.append("")

    if preserve:
        lines.append(f"## Preserve These ({len(preserve)} passages)\n")
        lines.append("These look intentional — do NOT merge or rewrite:\n")
        for i in preserve:
            lines.append(f"  [{i.paragraph_idx}] {i.reason}: \"{i.sentence_text}\"")

    return "\n".join(lines)
