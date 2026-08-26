from pathlib import Path

from classifier.extraction import classify
from classifier.schema import ClassificationResult, MarkScheme, StepClassification, StudentStep

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_scheme(name: str) -> MarkScheme:
    return MarkScheme.model_validate_json((DATA_DIR / "mark_schemes" / name).read_text())


def _knowledge_gap_resolver(scheme, student_id, steps) -> ClassificationResult:
    """Stand-in for the LLM fallback: always says knowledge_gap, tagged source=llm.

    Lets extraction.py's routing logic be tested without an API key.
    """
    return ClassificationResult(
        question_id=scheme.question_id,
        student_id=student_id,
        steps=[
            StepClassification(
                step_index=s["step_index"],
                content=s["content"],
                label="knowledge_gap",
                justification="test resolver",
                source="llm",
            )
            for s in steps
        ],
    )


def test_all_correct_resolved_by_rules_alone():
    scheme = _load_scheme("simultaneous_equations.json")
    steps = [
        StudentStep(step_index=1, content="3x=12"),
        StudentStep(step_index=2, content="x=4"),
        StudentStep(step_index=3, content="y=2"),
    ]
    result = classify(scheme, "s1", steps, resolver=_knowledge_gap_resolver)
    assert [s.label for s in result.steps] == ["correct", "correct", "correct"]
    assert all(s.source == "rule" for s in result.steps)


def test_known_process_error_then_self_correction():
    scheme = _load_scheme("simultaneous_equations.json")
    steps = [
        StudentStep(step_index=1, content="3x=8"),
        StudentStep(step_index=2, content="3x=12"),
        StudentStep(step_index=3, content="x=4"),
        StudentStep(step_index=4, content="y=2"),
    ]
    result = classify(scheme, "s2", steps, resolver=_knowledge_gap_resolver)
    labels = {s.step_index: s.label for s in result.steps}
    assert labels == {1: "process_error", 2: "correct", 3: "correct", 4: "correct"}


def test_unresolved_step_is_routed_to_resolver():
    scheme = _load_scheme("quadratic_factorising.json")
    steps = [StudentStep(step_index=1, content="x^2=15-2x")]
    result = classify(scheme, "s3", steps, resolver=_knowledge_gap_resolver)
    assert result.steps[0].label == "knowledge_gap"
    assert result.steps[0].source == "llm"


def test_known_error_pattern_then_correct_on_quadratic_scheme():
    scheme = _load_scheme("quadratic_factorising.json")
    steps = [
        StudentStep(step_index=1, content="(x-5)(x+3)=0"),
        StudentStep(step_index=2, content="(x+5)(x-3)=0"),
        StudentStep(step_index=3, content="x=-5,x=3"),
    ]
    result = classify(scheme, "s4", steps, resolver=_knowledge_gap_resolver)
    assert [s.label for s in result.steps] == ["process_error", "correct", "correct"]


def test_normalization_ignores_case_and_whitespace():
    scheme = _load_scheme("simultaneous_equations.json")
    steps = [StudentStep(step_index=1, content="  3X = 12 ")]
    result = classify(scheme, "s5", steps, resolver=_knowledge_gap_resolver)
    assert result.steps[0].label == "correct"
    assert result.steps[0].criterion == "M1"
