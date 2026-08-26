"""Shared data model for the Lumina diagnostic classifier.

Mirrors the (question, mark scheme, student working) -> per-step label
shape from lumina-classifier-design-plan.md Step 3, generalized beyond the
exact-match-only mark schemes in lumina-live-mvp.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Label = Literal["correct", "process_error", "knowledge_gap", "valid_alternative"]
Source = Literal["rule", "llm"]


class ErrorPattern(BaseModel):
    match: str
    reason: str


class Criterion(BaseModel):
    code: str  # e.g. "M1", "A1", "B1"
    description: str
    expected_forms: list[str] = Field(min_length=1)
    error_patterns: list[ErrorPattern] = Field(default_factory=list)


class MarkSchemePath(BaseModel):
    path_id: str
    label: str
    steps: list[Criterion] = Field(min_length=1)


class MarkScheme(BaseModel):
    question_id: str
    question_text: str
    topic: str
    total_marks: int
    paths: list[MarkSchemePath] = Field(min_length=1)


class StudentStep(BaseModel):
    step_index: int
    content: str


class StepClassification(BaseModel):
    step_index: int
    content: str
    label: Label
    criterion: Optional[str] = None
    justification: str
    source: Source = "rule"


class ClassificationResult(BaseModel):
    question_id: str
    student_id: str
    steps: list[StepClassification]


class GoldExample(BaseModel):
    question_id: str
    student_id: str
    steps: list[StudentStep]
    expected: list[StepClassification]
