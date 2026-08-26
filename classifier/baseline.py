"""Option C: prompted-LLM baseline classifier.

Per lumina-classifier-design-plan.md Step 2 (Option C) — this is the
always-needed baseline that Option B's rule layer falls back to for
unresolved steps, and that any future Option A fine-tune must beat on the
eval harness before it ships.

One call classifies every step for a question at once (not one call per
step): the design doc's hard case is disambiguating a process error from a
knowledge gap, which often requires the rest of the student's working for
context, not just the failing step in isolation.
"""

from __future__ import annotations

import json
import os

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from classifier import rate_limit
from classifier.schema import ClassificationResult, MarkScheme, StepClassification

# Loaded here because every LLM-calling entrypoint (extraction.py's
# fallback, distill.py, run_eval.py) imports this module — one call
# covers all of them. Pulls ANTHROPIC_API_KEY from a local, gitignored
# .env file; never put the real key in a tracked file.
load_dotenv()

MODEL = "claude-opus-5"

SYSTEM_TEMPLATE = """\
You are classifying a GCSE maths student's step-by-step working against an \
official mark scheme.

For each step, assign exactly one label:
- "correct": satisfies the next mark-scheme criterion on some valid method \
path (use "valid_alternative" instead if it's a valid path other than the \
one the student has been following).
- "process_error": the student understood the correct method for this step \
but made a procedural slip (arithmetic error, sign error, mis-substitution, \
mis-copied value). The method shown is recognizably the right one, just \
executed badly.
- "knowledge_gap": the step doesn't correspond to any valid method path in \
the mark scheme, and isn't explainable as a slip on a valid method — the \
student appears not to understand what method this step of the question \
requires.

Disambiguating process_error from knowledge_gap is the hard part: judge it \
using the *rest* of the student's working, not the failing step alone. A \
wrong substitution consistent with an otherwise-correct method is a slip; \
a step that reflects a fundamentally different (wrong) approach is a gap.

Question: {question_text}

Mark scheme (JSON, one or more valid method paths):
{mark_scheme_json}

For every step, give a one-sentence justification a teacher could read and \
trust. Cite the criterion code the step relates to when applicable.
"""


class _ClassificationBatch(BaseModel):
    steps: list[StepClassification]


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def classify(
    scheme: MarkScheme,
    student_id: str,
    steps: list[dict],
    client: anthropic.Anthropic | None = None,
) -> ClassificationResult:
    """Classify every step of one student's working against `scheme`.

    `steps` is a list of {"step_index": int, "content": str} dicts, in order.
    """
    client = client or _client()

    system_text = SYSTEM_TEMPLATE.format(
        question_text=scheme.question_text,
        mark_scheme_json=scheme.model_dump_json(indent=2),
    )

    user_content = "Student's working, in order:\n" + "\n".join(
        f"Step {s['step_index']}: {s['content']}" for s in steps
    )

    rate_limit.default_limiter().acquire()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_content}],
        output_format=_ClassificationBatch,
    )

    batch = response.parsed_output
    for step in batch.steps:
        step.source = "llm"

    return ClassificationResult(
        question_id=scheme.question_id,
        student_id=student_id,
        steps=batch.steps,
    )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Classify one student's working against a mark scheme.")
    parser.add_argument("--mark-scheme", required=True, help="Path to a mark scheme JSON file")
    parser.add_argument("--student-id", default="demo-student")
    parser.add_argument(
        "--steps",
        required=True,
        help='JSON list of {"step_index": int, "content": str}, e.g. \'[{"step_index":1,"content":"2x+y=10"}]\'',
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set — see .env.example.")

    scheme = MarkScheme.model_validate_json(Path(args.mark_scheme).read_text())
    result = classify(scheme, args.student_id, json.loads(args.steps))
    print(result.model_dump_json(indent=2))
