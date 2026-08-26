"""Run a classifier engine against a gold-labelled JSONL set and report metrics.

Engines:
  rule    - extraction.py's deterministic rule layer only; steps it can't
            resolve are labelled knowledge_gap without an LLM call.
  hybrid  - extraction.py with its default LLM fallback for unresolved
            steps (Option B as specified for the pilot).
  llm     - baseline.py alone, no rule layer (Option C).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from classifier.baseline import classify as baseline_classify
from classifier.eval.metrics import compute_metrics, format_report
from classifier.extraction import classify as extraction_classify
from classifier.schema import ClassificationResult, GoldExample, MarkScheme, StepClassification

MARK_SCHEME_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "mark_schemes"


def _load_mark_schemes() -> dict[str, MarkScheme]:
    schemes = {}
    for path in MARK_SCHEME_DIR.glob("*.json"):
        scheme = MarkScheme.model_validate_json(path.read_text())
        schemes[scheme.question_id] = scheme
    return schemes


def _rule_only_resolver(scheme: MarkScheme, student_id: str, steps: list[dict]) -> ClassificationResult:
    return ClassificationResult(
        question_id=scheme.question_id,
        student_id=student_id,
        steps=[
            StepClassification(
                step_index=s["step_index"],
                content=s["content"],
                label="knowledge_gap",
                justification="Unresolved by the rule layer; rule-only mode does not call the LLM fallback.",
                source="rule",
            )
            for s in steps
        ],
    )


def load_gold(path: Path) -> list[GoldExample]:
    examples = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        examples.append(GoldExample.model_validate_json(line))
    return examples


def run(engine: str, gold_path: Path) -> tuple[list[tuple[str, str]], list[str]]:
    schemes = _load_mark_schemes()
    examples = load_gold(gold_path)

    pairs: list[tuple[str, str]] = []
    mismatches: list[str] = []

    for example in examples:
        scheme = schemes[example.question_id]

        if engine == "rule":
            result = extraction_classify(scheme, example.student_id, example.steps, resolver=_rule_only_resolver)
        elif engine == "hybrid":
            result = extraction_classify(scheme, example.student_id, example.steps)
        elif engine == "llm":
            steps_payload = [{"step_index": s.step_index, "content": s.content} for s in example.steps]
            result = baseline_classify(scheme, example.student_id, steps_payload)
        else:
            raise ValueError(f"Unknown engine: {engine}")

        predicted_by_index = {sc.step_index: sc.label for sc in result.steps}
        for expected in example.expected:
            predicted = predicted_by_index.get(expected.step_index, "MISSING")
            pairs.append((expected.label, predicted))
            if predicted != expected.label:
                mismatches.append(
                    f"  {example.question_id}/{example.student_id} step {expected.step_index}: "
                    f'gold={expected.label} predicted={predicted}  ("{expected.content}")'
                )

    return pairs, mismatches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a classifier engine against a gold set.")
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--engine", choices=["rule", "hybrid", "llm"], default="rule")
    args = parser.parse_args()

    if args.engine in ("hybrid", "llm") and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(f'--engine {args.engine} calls the LLM baseline; ANTHROPIC_API_KEY is not set.')

    pairs, mismatches = run(args.engine, args.gold)
    metrics = compute_metrics(pairs)
    print(format_report(metrics))
    if mismatches:
        print("\nmismatches:")
        print("\n".join(mismatches))
