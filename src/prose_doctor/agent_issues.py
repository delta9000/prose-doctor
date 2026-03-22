"""Per-passage issue finder for the revision agent.

Given a metric name and a chapter's text + report, returns specific
passages that are problematic — with paragraph indices, text excerpts,
and surrounding context. This lets the agent target edits precisely
instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from prose_doctor.critique_config import CritiqueConfig
from prose_doctor.lenses.fragment_classifier import _has_concrete_detail, _is_vague_fragment
from prose_doctor.providers import ProviderPool
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


def find_fragment_issues(text: str, report: dict, config: CritiqueConfig | None = None) -> list[Issue]:
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
    pool = ProviderPool()
    nlp = pool.spacy

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
                reason = (
                    "weak fragment — vague/generic content that adds nothing. "
                    "Either replace with a concrete sensory detail or fold "
                    "naturally into the adjacent sentence"
                )
                preserve = False
            elif is_concrete:
                reason = (
                    "fragment with concrete detail — may be intentional rhythm. "
                    "Leave it if it follows a long sentence or lands at an "
                    "emotional beat. Only merge if it feels random"
                )
                preserve = False
            else:
                reason = (
                    "isolated fragment — no concrete detail, no rhythmic purpose. "
                    "Fold into the adjacent sentence or strengthen with a specific "
                    "sensory/physical detail"
                )
                preserve = False

            issues.append(Issue(pi, sent_text, ctx_before, ctx_after, reason, preserve))

    return issues


def find_psychic_distance_issues(text: str, report: dict, config: CritiqueConfig | None = None) -> list[Issue]:
    """Find sentences where psychic distance is too external, with causes.

    Instead of just pointing at paragraphs, identifies specific sentences
    and explains WHY they're too distant (no perception verbs, no pronouns,
    expository tone, etc).
    """
    from prose_doctor.lenses.psychic_distance import (
        PERCEPTION_VERBS, COGNITION_VERBS, PROXIMAL_DEICTICS,
    )
    pool = ProviderPool()
    nlp = pool.spacy

    paragraphs = split_paragraphs(text)
    pd = report.get("psychic_distance") or {}
    para_means = pd.get("paragraph_means", [])

    cfg = config or CritiqueConfig()
    baseline = cfg.baselines["pd_mean"][0]
    issues = []

    for pi, mean in enumerate(para_means):
        if pi >= len(paragraphs):
            break
        if mean >= baseline - cfg.pd_baseline_margin:
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
            if len(causes) >= cfg.pd_cause_threshold:
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


def find_inversion_issues(text: str, report: dict, config: CritiqueConfig | None = None) -> list[Issue]:
    """Find specific SVO sentences that could be inverted, with context."""
    pool = ProviderPool()
    nlp = pool.spacy

    cfg = config or CritiqueConfig()
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

        if inv_pct < cfg.inversion_pct_gate and len(svo_sents) >= 3:
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


def find_flatline_issues(text: str, report: dict, config: CritiqueConfig | None = None) -> list[Issue]:
    """Find information flatlines with proper sentence-to-paragraph mapping."""
    pool = ProviderPool()
    nlp = pool.spacy

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


def find_spike_issues(text: str, report: dict, config: CritiqueConfig | None = None) -> list[Issue]:
    """Find paragraphs that are too predictable and need surprisal spikes.

    When a chapter has too few spikes, this identifies the flattest passages
    where injecting surprise would help most.
    """
    pool = ProviderPool()
    nlp = pool.spacy

    paragraphs = split_paragraphs(text)
    ic = report.get("info_contour") or {}
    surprisals = ic.get("sentence_surprisals", []) if hasattr(ic, 'get') else []

    cfg = config or CritiqueConfig()

    # If we don't have per-sentence data, fall back to flatlines
    if not surprisals:
        return find_flatline_issues(text, report, config=config)

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
    indexed = [(i, s) for i, s in enumerate(surprisals) if s < mean_s - cfg.spike_surprisal_margin]
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


def find_generic_issues(text: str, report: dict, config: CritiqueConfig | None = None) -> list[Issue]:
    """Find structurally generic paragraphs with actionable decomposition.

    Uses narrative attention feature decomposition to identify paragraphs
    that are structurally indistinguishable from the chapter average, and
    explains which specific features to change.

    Lightweight version: reuses scan data from report, only adds cheap
    Boyd function-word features and verb energy (no GPT-2, no emotion model).
    """
    import numpy as np
    pool = ProviderPool()
    nlp = pool.spacy

    paragraphs = split_paragraphs(text)
    n = len(paragraphs)
    if n < 10:
        return []

    pd = report.get("psychic_distance") or {}
    fg = report.get("foregrounding") or {}
    ic = report.get("info_contour") or {}
    para_means = pd.get("paragraph_means", [])

    # Boyd function word sets
    _STAGING = frozenset({
        "in", "on", "at", "to", "from", "by", "with", "into", "through",
        "across", "between", "among", "above", "below", "beneath", "behind",
        "beside", "near", "toward", "towards", "along", "around", "against",
        "over", "under", "within", "without", "upon", "during", "before",
        "after", "until", "beyond", "past", "outside", "inside",
        "the", "a", "an",
    })
    _PROGRESSION = frozenset({
        "would", "could", "should", "might", "must", "shall", "will",
        "can", "may", "do", "does", "did", "has", "have", "had",
        "was", "were", "been", "being", "am", "is", "are",
        "then", "now", "just", "still", "already", "again", "soon",
        "quickly", "slowly", "suddenly", "finally", "immediately",
    })
    _TENSION = frozenset({
        "think", "thought", "know", "knew", "believe", "believed",
        "understand", "understood", "realize", "realized", "wonder",
        "wondered", "consider", "considered", "imagine", "imagined",
        "remember", "remembered", "decide", "decided",
        "because", "reason", "why", "whether", "if", "unless",
        "although", "though", "however", "but", "yet",
        "maybe", "perhaps", "probably",
        "right", "wrong", "true", "certain", "possible",
    })
    _DYNAMIC = frozenset({
        "ran", "run", "walked", "grabbed", "pulled", "pushed", "threw",
        "hit", "jumped", "climbed", "fell", "opened", "closed", "turned",
        "moved", "reached", "picked", "dropped", "kicked", "slammed",
        "ripped", "cut", "shot", "fired", "swung", "caught",
        "lunged", "stumbled", "sprinted", "crawled", "lifted", "carried",
    })
    _STATIVE = frozenset({
        "was", "were", "is", "seemed", "appeared", "felt", "looked",
        "remained", "stayed", "stood", "sat", "lay", "had", "knew",
        "thought", "believed", "wanted", "needed",
    })

    # Build per-paragraph features
    features = []
    feature_names = [
        "pd_mean", "fragment_ratio", "inversion_ratio", "sl_cv",
        "dialogue_ratio", "verb_energy",
        "boyd_staging", "boyd_progression", "boyd_tension",
    ]

    for pi, para in enumerate(paragraphs):
        words = para.lower().split()
        wc = len(words) or 1
        word_set = set(words)

        doc = nlp(para)
        sents = list(doc.sents)

        # Psychic distance (from report)
        pd_val = para_means[pi] if pi < len(para_means) else 0.3

        # Fragment ratio
        frag = sum(1 for s in sents if len(s) < 5) / max(len(sents), 1) if sents else 0

        # Inversion
        inv = 0
        total_s = 0
        for sent in sents:
            total_s += 1
            root = None
            for t in sent:
                if t.dep_ == "ROOT":
                    root = t
                if t.dep_ in ("nsubj", "nsubjpass") and root is not None and t.i > root.i:
                    inv += 1
                    break
        inv_ratio = inv / max(total_s, 1)

        # Sentence length CV
        lengths = [len(s) for s in sents if len(s) > 2]
        sl_cv = float(np.std(lengths) / np.mean(lengths)) if len(lengths) >= 2 else 0

        # Dialogue ratio
        dial_chars = sum(len(l) for l in para.split('\n') if '"' in l or '\u201c' in l)
        dial_ratio = dial_chars / max(len(para), 1)

        # Verb energy
        verbs = [t.text.lower() for t in doc if t.pos_ == "VERB"]
        if verbs:
            dyn = sum(1 for v in verbs if v in _DYNAMIC)
            sta = sum(1 for v in verbs if v in _STATIVE)
            ve = dyn / max(dyn + sta, 1)
        else:
            ve = 0.5

        # Boyd features
        staging = len(word_set & _STAGING) / wc
        progression = len(word_set & _PROGRESSION) / wc
        tension = len(word_set & _TENSION) / wc

        features.append([pd_val, frag, inv_ratio, sl_cv, dial_ratio, ve,
                         staging, progression, tension])

    features_arr = np.array(features)

    # Normalize to [0,1]
    mins = features_arr.min(axis=0, keepdims=True)
    maxs = features_arr.max(axis=0, keepdims=True)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0
    normed = (features_arr - mins) / ranges

    # Chapter mean
    chapter_mean = normed.mean(axis=0)

    # Per-paragraph deviation from mean
    deviations = np.abs(normed - chapter_mean)
    generic_scores = deviations.mean(axis=1)

    # Find the most generic paragraphs
    ranking = np.argsort(generic_scores)
    issues = []

    for idx in ranking[:12]:
        devs = deviations[idx]
        feat_order = np.argsort(devs)

        # Build prescription from the most average features
        avg_features = []
        for fi in feat_order[:3]:
            avg_features.append(feature_names[fi])

        distinct_features = []
        for fi in feat_order[-2:]:
            distinct_features.append(feature_names[fi])

        # Human-readable prescriptions
        prescriptions = {
            "pd_mean": "shift psychic distance — go deeper into interiority or pull back to establishing shot",
            "fragment_ratio": "change sentence structure — add fragments for tension or merge them for flow",
            "inversion_ratio": "restructure sentences — invert word order to break SVO monotony",
            "sl_cv": "vary sentence lengths — mix short punches with long flowing sentences",
            "dialogue_ratio": "add or remove dialogue to shift the paragraph's mode",
            "verb_energy": "change verb energy — use dynamic/action verbs or commit to stative/perceptive ones",
            "boyd_staging": "adjust world-building language — add or remove spatial/temporal grounding",
            "boyd_progression": "adjust plot-moving language — add or remove action modifiers and auxiliaries",
            "boyd_tension": "adjust cognitive tension — add reasoning, doubt, evaluation, or remove it",
        }

        fix_suggestions = [prescriptions.get(f, f) for f in avg_features]
        preserve_notes = [f"{f} (already distinctive)" for f in distinct_features]

        reason = (
            f"structurally generic — most average in: {', '.join(avg_features)}. "
            f"Fix: {fix_suggestions[0]}"
        )

        issues.append(Issue(
            paragraph_idx=int(idx),
            sentence_text=paragraphs[idx][:150],
            context_before="",
            context_after=f"Preserve: {', '.join(preserve_notes)}",
            reason=reason,
            preserve=False,
        ))

    return issues


def find_discourse_issues(text: str, report: dict, config: CritiqueConfig | None = None) -> list[Issue]:
    """Find paragraphs with monotonous discourse relations.

    Targets two patterns empirically shown to separate AI from human prose:
    1. Long runs of implicit relations (no connectives at all)
    2. Additive-only zones (only "and" connectives, no causal/contrastive)

    Prescriptions focus on adding causal ("because", "so") and contrastive
    ("but", "however") connectives — the types most underused by AI.
    """
    pool = ProviderPool()
    nlp = pool.spacy

    paragraphs = split_paragraphs(text)
    dr = report.get("discourse_relations") or {}
    entropy = dr.get("relation_entropy", 1.0)
    implicit = dr.get("implicit_ratio", 0.0)

    cfg = config or CritiqueConfig()

    # Only flag if chapter-level metrics are in AI territory.
    if entropy > cfg.discourse_entropy_gate and implicit < cfg.discourse_implicit_gate:
        return []

    from prose_doctor.lenses.discourse_relations import _classify_sentence

    issues = []
    consecutive_implicit = 0

    for pi, para in enumerate(paragraphs):
        doc = nlp(para)
        sents = list(doc.sents)
        para_relations = []

        for sent in sents:
            rel_name, _, evidence = _classify_sentence(sent.text)
            para_relations.append(rel_name)

        # Count implicit sentences in this paragraph
        n_implicit = sum(1 for r in para_relations if r == "implicit")
        n_additive = sum(1 for r in para_relations if r == "additive")
        n_total = len(para_relations)

        if n_total < 2:
            consecutive_implicit = 0
            continue

        all_implicit = n_implicit == n_total
        all_additive_or_implicit = all(r in ("implicit", "additive") for r in para_relations)

        if all_implicit:
            consecutive_implicit += 1
        else:
            consecutive_implicit = 0

        # Flag: 3+ consecutive all-implicit paragraphs
        if consecutive_implicit >= cfg.consecutive_implicit_trigger:
            first_sent = sents[0].text.strip()[:150] if sents else ""
            ctx_before = paragraphs[pi - 1][:100] if pi > 0 else ""
            ctx_after = paragraphs[pi + 1][:100] if pi < len(paragraphs) - 1 else ""
            issues.append(Issue(
                paragraph_idx=pi,
                sentence_text=first_sent,
                context_before=ctx_before,
                context_after=ctx_after,
                reason=(
                    f"implicit relation run ({consecutive_implicit} paragraphs with no connectives) — "
                    f"add a causal ('because', 'since') or contrastive ('but', 'however') "
                    f"connective to show how sentences relate"
                ),
                preserve=False,
            ))

        # Flag: additive-only paragraph (all "and" connectives)
        elif all_additive_or_implicit and n_additive >= cfg.additive_count_trigger:
            first_sent = sents[0].text.strip()[:150] if sents else ""
            ctx_before = paragraphs[pi - 1][:100] if pi > 0 else ""
            ctx_after = paragraphs[pi + 1][:100] if pi < len(paragraphs) - 1 else ""
            issues.append(Issue(
                paragraph_idx=pi,
                sentence_text=first_sent,
                context_before=ctx_before,
                context_after=ctx_after,
                reason=(
                    f"additive-only paragraph ({n_additive} 'and' connectives, no causal or contrastive) — "
                    f"replace some 'and' with 'because', 'so', 'but', or 'although' "
                    f"to show logical relationships"
                ),
                preserve=False,
            ))

    return issues[:15]


def find_concreteness_issues(text: str, report: dict, config: CritiqueConfig | None = None) -> list[Issue]:
    """Find passages that are excessively concrete or use vague nouns.

    Two complementary signals:
    1. Chapters with very low abstractness_ratio (< 0.15) need moments of
       reflection, interpretation, or generalization. AI prose tends to
       stay relentlessly concrete.
    2. Vague nouns ("thing", "stuff", "something") that could be replaced
       with specific terms.
    """
    pool = ProviderPool()
    nlp = pool.spacy

    paragraphs = split_paragraphs(text)
    cn = report.get("concreteness") or {}
    abstract_ratio = cn.get("abstractness_ratio", 0.5)
    vague_density = cn.get("vague_noun_density", 0)

    cfg = config or CritiqueConfig()
    issues = []

    from prose_doctor.lenses.concreteness import VAGUE_NOUNS

    # Issue type 1: Vague nouns
    if vague_density > cfg.vague_density_gate:
        for pi, para in enumerate(paragraphs):
            doc = nlp(para)
            for sent in doc.sents:
                for token in sent:
                    if token.lemma_.lower() in VAGUE_NOUNS:
                        sent_text = sent.text.strip()[:150]
                        ctx_before = paragraphs[pi - 1][:100] if pi > 0 else ""
                        ctx_after = paragraphs[pi + 1][:100] if pi < len(paragraphs) - 1 else ""
                        issues.append(Issue(
                            paragraph_idx=pi,
                            sentence_text=sent_text,
                            context_before=ctx_before,
                            context_after=ctx_after,
                            reason=(
                                f"vague noun '{token.text}' — replace with a specific noun "
                                f"(what thing? what kind? what exactly?)"
                            ),
                            preserve=False,
                        ))
                        break  # one per sentence

    # Issue type 2: No abstraction — flag longest concrete-only stretches
    if abstract_ratio < cfg.abstract_ratio_gate:
        from prose_doctor.lenses.concreteness import _load_norms
        norms = _load_norms()

        concrete_run = 0
        for pi, para in enumerate(paragraphs):
            doc = nlp(para)
            scores = []
            for token in doc:
                if not token.is_alpha or token.is_stop or len(token.text) <= 2:
                    continue
                word = token.lemma_.lower()
                if word in norms:
                    scores.append(norms[word])

            para_mean = sum(scores) / len(scores) if scores else 3.0
            if para_mean > cfg.concrete_para_mean_threshold:
                concrete_run += 1
            else:
                concrete_run = 0

            if concrete_run >= cfg.concrete_run_trigger:
                sent_text = paragraphs[pi][:150]
                ctx_before = paragraphs[pi - 1][:100] if pi > 0 else ""
                ctx_after = paragraphs[pi + 1][:100] if pi < len(paragraphs) - 1 else ""
                issues.append(Issue(
                    paragraph_idx=pi,
                    sentence_text=sent_text,
                    context_before=ctx_before,
                    context_after=ctx_after,
                    reason=(
                        f"concrete run ({concrete_run} paragraphs with no abstraction) — "
                        f"add a moment of reflection, interpretation, memory, or opinion "
                        f"to break the sensory surface"
                    ),
                    preserve=False,
                ))

    return issues[:15]


def find_shift_issues(text: str, report: dict, config: CritiqueConfig | None = None) -> list[Issue]:
    """Find long static stretches that lack time, space, or character shifts.

    Human prose averages ~1.5-2 shifts per paragraph. AI prose averages ~1.
    Flag runs of 5+ paragraphs with no shifts — these are the most static
    stretches where a cut, time skip, or new character would add dynamism.
    """
    paragraphs = split_paragraphs(text)
    ss = report.get("situation_shifts") or {}
    total = ss.get("total_shifts", 999)
    n_paras = max(len(paragraphs), 1)
    shift_rate = total / n_paras

    cfg = config or CritiqueConfig()

    # Only flag if shift rate is below human baseline
    if shift_rate > cfg.shift_rate_gate:
        return []

    pool = ProviderPool()
    nlp = pool.spacy

    from prose_doctor.lenses.situation_shifts import (
        _has_temporal_markers, _get_first_subject,
    )

    issues = []
    no_shift_run = 0
    prev_subject = None

    for pi, para in enumerate(paragraphs):
        doc = nlp(para)
        cur_subject = _get_first_subject(doc)
        has_time = _has_temporal_markers(para)

        has_any_shift = has_time  # time marker = shift
        if cur_subject and prev_subject and cur_subject != prev_subject:
            if cur_subject not in ("he", "she", "they", "it") and prev_subject not in ("he", "she", "they", "it"):
                has_any_shift = True

        if has_any_shift:
            no_shift_run = 0
        else:
            no_shift_run += 1

        if no_shift_run >= cfg.no_shift_run_trigger:
            issues.append(Issue(
                paragraph_idx=pi,
                sentence_text=paragraphs[pi][:150],
                context_before=paragraphs[pi - 1][:100] if pi > 0 else "",
                context_after=paragraphs[pi + 1][:100] if pi < len(paragraphs) - 1 else "",
                reason=(
                    f"static scene ({no_shift_run} paragraphs, same time/place/character) — "
                    f"consider a time reference, location change, new character entering, "
                    f"or a scene cut"
                ),
                preserve=False,
            ))

        prev_subject = cur_subject

    return issues[:10]


METRIC_FINDERS = {
    "fg_fragment": find_fragment_issues,
    "fg_inversion": find_inversion_issues,
    "pd_mean": find_psychic_distance_issues,
    "pd_std": find_psychic_distance_issues,
    "ic_rhythmicity": find_flatline_issues,
    "ic_flatlines": find_flatline_issues,
    "ic_spikes": find_spike_issues,
    "generic": find_generic_issues,
    "dr_entropy": find_discourse_issues,
    "dr_implicit": find_discourse_issues,
    "cn_abstract": find_concreteness_issues,
    "ss_shift_rate": find_shift_issues,
}


def find_issues(metric: str, text: str, report: dict, config: CritiqueConfig | None = None) -> list[Issue]:
    """Find specific problematic passages for a metric."""
    finder = METRIC_FINDERS.get(metric)
    if finder is None:
        return []
    return finder(text, report, config=config)


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
