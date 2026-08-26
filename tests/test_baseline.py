import os
from pathlib import Path

import pytest

from classifier.baseline import classify
from classifier.schema import MarkScheme

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="requires ANTHROPIC_API_KEY"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_baseline_classifies_an_all_correct_submission():
    scheme = MarkScheme.model_validate_json(
        (DATA_DIR / "mark_schemes" / "simultaneous_equations.json").read_text()
    )
    steps = [
        {"step_index": 1, "content": "3x=12"},
        {"step_index": 2, "content": "x=4"},
        {"step_index": 3, "content": "y=2"},
    ]
    result = classify(scheme, "live-test", steps)
    assert len(result.steps) == 3
    assert all(s.label == "correct" for s in result.steps)
    assert all(s.source == "llm" for s in result.steps)
