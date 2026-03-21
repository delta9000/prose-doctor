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


def find_fragment_issues(text: str, report: dict) -> list[Issue]:
    """Find specific fragment sentences and classify as crutch vs intentional.

    Fragments are sentences with < 5 tokens. Classification heuristics:
    - In a sequence of 3+ fragments: likely a deliberate list/montage → preserve
    - Fragment containing dialogue punctuation: likely dialogue tag → preserve
    - Isolated fragment after a long sentence: likely emphasis crutch → fix
    - Fragment that echoes/mirrors another fragment nearby: likely deliberate → preserve
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

            # Context
            ctx_before = sents[si - 1].text.strip() if si > 0 else ""
            ctx_after = sents[si + 1].text.strip() if si < len(sents) - 1 else ""

            # Classification heuristics
            preserve = False
            reason = "isolated fragment — likely emphasis crutch"

            # Check: part of a fragment run (3+ consecutive short sentences)?
            run_start = si
            while run_start > 0 and sent_lengths[run_start - 1] < 5:
                run_start -= 1
            run_end = si
            while run_end < len(sents) - 1 and sent_lengths[run_end + 1] < 5:
                run_end += 1
            run_length = run_end - run_start + 1

            if run_length >= 2:
                preserve = True
                if run_length >= 3:
                    reason = f"part of {run_length}-fragment sequence — likely deliberate montage/list"
                else:
                    reason = "fragment pair — likely deliberate contrast/echo"

            # Check: anaphoric echo (shares first word with adjacent fragment)?
            if si > 0 and sent_lengths[si - 1] < 5:
                prev_first = sents[si - 1].text.strip().split()[0].lower() if sents[si - 1].text.strip() else ""
                curr_first = sent_text.split()[0].lower() if sent_text else ""
                if prev_first and curr_first and prev_first == curr_first:
                    preserve = True
                    reason = f"anaphoric echo ('{curr_first}...') — deliberate repetition"

            # Check: contains dialogue markers?
            if '"' in sent_text or '"' in sent_text or '"' in sent_text:
                preserve = True
                reason = "dialogue fragment — preserve"

            # Check: contains em-dash (continuation)?
            if '—' in sent_text or '--' in sent_text:
                preserve = True
                reason = "em-dash continuation — preserve"

            # Check: follows a very long sentence (dramatic contrast)?
            if si > 0 and sent_lengths[si - 1] > 25:
                # Long sentence followed by very short = likely intentional rhythm
                # BUT only if it's not a pattern repeated everywhere
                preserve = True
                reason = "short after long sentence — likely intentional rhythm contrast"

            issues.append(Issue(
                paragraph_idx=pi,
                sentence_text=sent_text,
                context_before=ctx_before[:120] if ctx_before else "",
                context_after=ctx_after[:120] if ctx_after else "",
                reason=reason,
                preserve=preserve,
            ))

    return issues


def find_psychic_distance_issues(text: str, report: dict) -> list[Issue]:
    """Find paragraphs where psychic distance is far from baseline."""
    paragraphs = split_paragraphs(text)
    pd = report.get("psychic_distance") or {}
    para_means = pd.get("paragraph_means", [])

    baseline = 0.336  # human baseline mean
    issues = []

    for pi, mean in enumerate(para_means):
        if pi >= len(paragraphs):
            break
        gap = baseline - mean
        if gap > 0.1:  # significantly below baseline (too distant)
            para_text = paragraphs[pi]
            issues.append(Issue(
                paragraph_idx=pi,
                sentence_text=para_text[:200],
                context_before="",
                context_after="",
                reason=f"psychic distance {mean:.3f} (baseline: {baseline}) — "
                       f"too distant/external, needs interiority",
                preserve=False,
            ))

    # Sort by gap (worst first)
    issues.sort(key=lambda i: float(i.reason.split()[2]))
    return issues


def find_inversion_issues(text: str, report: dict) -> list[Issue]:
    """Find paragraphs with zero or very low sentence inversion."""
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

        # Count inversions in this paragraph
        inversions = 0
        svo_sents = []
        for sent in sents:
            root = None
            for t in sent:
                if t.dep_ == "ROOT":
                    root = t
                if t.dep_ in ("nsubj", "nsubjpass") and root is not None and t.i > root.i:
                    inversions += 1
                    break
            else:
                svo_sents.append(sent.text.strip()[:100])

        inv_pct = inversions / total * 100
        if inv_pct < 10 and total >= 4:
            # All SVO — suggest restructuring
            example_svo = svo_sents[0] if svo_sents else ""
            issues.append(Issue(
                paragraph_idx=pi,
                sentence_text=example_svo,
                context_before="",
                context_after="",
                reason=f"0/{total} inverted sentences — monotonous SVO order",
                preserve=False,
            ))

    return issues


def find_flatline_issues(text: str, report: dict) -> list[Issue]:
    """Find information flatlines — stretches of uniform surprisal."""
    paragraphs = split_paragraphs(text)
    ic = report.get("info_contour") or {}
    flatlines = ic.get("flatlines", [])

    if isinstance(flatlines, int):
        return []

    issues = []
    for fl in flatlines:
        if not isinstance(fl, dict):
            continue
        start = fl.get("start", 0)
        end = fl.get("end", start)
        # Map sentence indices to approximate paragraph indices
        para_idx = min(start // 3, len(paragraphs) - 1)
        if para_idx < len(paragraphs):
            issues.append(Issue(
                paragraph_idx=para_idx,
                sentence_text=paragraphs[para_idx][:200],
                context_before="",
                context_after="",
                reason=f"information flatline (sentences {start}-{end}, "
                       f"mean surprisal {fl.get('mean_surprisal', 0):.3f}) — "
                       f"too predictable, break with unexpected detail or shift",
                preserve=False,
            ))

    return issues


METRIC_FINDERS = {
    "fg_fragment": find_fragment_issues,
    "fg_inversion": find_inversion_issues,
    "pd_mean": find_psychic_distance_issues,
    "pd_std": find_psychic_distance_issues,
    "ic_rhythmicity": find_flatline_issues,
    "ic_flatlines": find_flatline_issues,
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
