"""Option B: structured extraction + rule layer + targeted LLM fallback.

Per lumina-classifier-design-plan.md Step 2 (Option B) — the version meant
to ship for the pilot. Exact-match rule logic here is a Python port of the
proven matcher in lumina-live-mvp/src/lib/diagnosis.ts (normalize -> check
current path's next criterion -> check other paths' next criterion -> check
known error patterns), generalized to run over a full submission at once
rather than one live step at a time.

Unlike the demo's matcher, a step that resolves against none of the above
is not dumped into a single "method-divergence" bucket: every unresolved
step for a submission is batched into one call to the Option C baseline
classifier (which sees the full ordered working, not just the failing
step, per the design doc's guidance on disambiguating process error from
knowledge gap), and only the LLM's process_error/knowledge_gap verdict is
kept for those steps — rule-resolved steps are left as-is, since the rule
layer is deterministic and the design doc treats that as the reliable,
interpretable path wherever it applies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from classifier.schema import (
    ClassificationResult,
    Criterion,
    MarkScheme,
    MarkSchemePath,
    StepClassification,
    StudentStep,
)

Resolver = Callable[[MarkScheme, str, list[dict]], ClassificationResult]


def normalize(raw: str) -> str:
    s = raw.lower()
    s = s.replace("×", "*").replace("÷", "/")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\.$", "", s)
    return s


def _step_matches(criterion: Criterion, normalized: str) -> bool:
    return normalized in {normalize(f) for f in criterion.expected_forms}


def _find_path(scheme: MarkScheme, path_id: str) -> MarkSchemePath:
    for path in scheme.paths:
        if path.path_id == path_id:
            return path
    raise KeyError(f'Unknown path_id "{path_id}" for question "{scheme.question_id}"')


@dataclass
class _Cursor:
    path_id: str
    next_step_index: int = 0


@dataclass
class _State:
    cursors: dict[str, _Cursor] = field(default_factory=dict)
    primary_path_id: Optional[str] = None


def classify(
    scheme: MarkScheme,
    student_id: str,
    steps: list[StudentStep],
    resolver: Resolver | None = None,
) -> ClassificationResult:
    if resolver is None:
        from classifier.baseline import classify as _baseline_classify

        resolver = _baseline_classify

    state = _State(cursors={p.path_id: _Cursor(p.path_id) for p in scheme.paths})
    resolved: dict[int, StepClassification] = {}
    unresolved: list[StudentStep] = []

    for step in steps:
        normalized = normalize(step.content)
        matched = _try_primary_path(scheme, state, step, normalized, resolved)
        if not matched:
            matched = _try_alternative_path(scheme, state, step, normalized, resolved)
        if not matched:
            matched = _try_error_pattern(scheme, state, step, normalized, resolved)
        if not matched:
            unresolved.append(step)

    if unresolved:
        all_steps_payload = [{"step_index": s.step_index, "content": s.content} for s in steps]
        llm_result = resolver(scheme, student_id, all_steps_payload)
        llm_by_index = {sc.step_index: sc for sc in llm_result.steps}
        for step in unresolved:
            sc = llm_by_index.get(step.step_index)
            if sc is None:
                sc = StepClassification(
                    step_index=step.step_index,
                    content=step.content,
                    label="knowledge_gap",
                    justification="No rule match, and the LLM resolver returned no classification for this step.",
                    source="llm",
                )
            else:
                sc.source = "llm"
            resolved[step.step_index] = sc

    ordered = sorted(resolved.values(), key=lambda sc: sc.step_index)
    return ClassificationResult(question_id=scheme.question_id, student_id=student_id, steps=ordered)


def _try_primary_path(scheme, state, step, normalized, resolved) -> bool:
    if state.primary_path_id is None:
        return False
    cursor = state.cursors[state.primary_path_id]
    path = _find_path(scheme, state.primary_path_id)
    if cursor.next_step_index >= len(path.steps):
        return False
    crit = path.steps[cursor.next_step_index]
    if not _step_matches(crit, normalized):
        return False
    resolved[step.step_index] = StepClassification(
        step_index=step.step_index,
        content=step.content,
        label="correct",
        criterion=crit.code,
        justification=f"Matches {crit.code}: {crit.description}.",
        source="rule",
    )
    cursor.next_step_index += 1
    return True


def _try_alternative_path(scheme, state, step, normalized, resolved) -> bool:
    for path in scheme.paths:
        if path.path_id == state.primary_path_id:
            continue
        cursor = state.cursors[path.path_id]
        if cursor.next_step_index >= len(path.steps):
            continue
        crit = path.steps[cursor.next_step_index]
        if not _step_matches(crit, normalized):
            continue
        label = "correct" if state.primary_path_id is None else "valid_alternative"
        note = (
            f"Matches {crit.code}: {crit.description}."
            if label == "correct"
            else f"Followed alternative method path: {path.label} ({crit.code})."
        )
        resolved[step.step_index] = StepClassification(
            step_index=step.step_index,
            content=step.content,
            label=label,
            criterion=crit.code,
            justification=note,
            source="rule",
        )
        cursor.next_step_index += 1
        state.primary_path_id = state.primary_path_id or path.path_id
        return True
    return False


def _try_error_pattern(scheme, state, step, normalized, resolved) -> bool:
    tracked_ids = [state.primary_path_id] if state.primary_path_id else [p.path_id for p in scheme.paths]
    for path_id in tracked_ids:
        path = _find_path(scheme, path_id)
        cursor = state.cursors[path_id]
        if cursor.next_step_index >= len(path.steps):
            continue
        crit = path.steps[cursor.next_step_index]
        for err in crit.error_patterns:
            if normalize(err.match) != normalized:
                continue
            resolved[step.step_index] = StepClassification(
                step_index=step.step_index,
                content=step.content,
                label="process_error",
                criterion=crit.code,
                justification=err.reason,
                source="rule",
            )
            state.primary_path_id = state.primary_path_id or path_id
            return True
    return False
