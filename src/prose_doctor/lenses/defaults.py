"""Default lens registry with all standard lenses."""
from prose_doctor.lenses import LensRegistry


def default_registry() -> LensRegistry:
    registry = LensRegistry()
    from prose_doctor.lenses.pacing import PacingLens
    from prose_doctor.lenses.emotion_arc import EmotionArcLens
    from prose_doctor.lenses.foregrounding import ForegroundingLens
    from prose_doctor.lenses.info_contour import InfoContourLens
    from prose_doctor.lenses.psychic_distance import PsychicDistanceLens
    from prose_doctor.lenses.sensory import SensoryLens
    from prose_doctor.lenses.dialogue_voice import DialogueVoiceLens
    from prose_doctor.lenses.slop_classifier import SlopClassifierLens
    from prose_doctor.lenses.perplexity import PerplexityLens
    from prose_doctor.lenses.uncertainty_reduction import UncertaintyReductionLens
    from prose_doctor.lenses.boyd_narrative_role import BoydNarrativeRoleLens
    from prose_doctor.lenses.fragment_classifier import FragmentClassifierLens
    from prose_doctor.lenses.narrative_attention import NarrativeAttentionLens
    from prose_doctor.lenses.concreteness import ConcretenessLens
    from prose_doctor.lenses.referential_cohesion import ReferentialCohesionLens
    from prose_doctor.lenses.situation_shifts import SituationShiftsLens
    from prose_doctor.lenses.discourse_relations import DiscourseRelationsLens

    for cls in [
        PacingLens, EmotionArcLens, ForegroundingLens, InfoContourLens,
        PsychicDistanceLens, SensoryLens, DialogueVoiceLens, SlopClassifierLens,
        PerplexityLens, UncertaintyReductionLens, BoydNarrativeRoleLens,
        FragmentClassifierLens, NarrativeAttentionLens,
        ConcretenessLens, ReferentialCohesionLens, SituationShiftsLens,
        DiscourseRelationsLens,
    ]:
        registry.register(cls())
    return registry
