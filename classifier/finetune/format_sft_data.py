"""Convert gold-labelled examples into SFT training pairs for Option A.

Per design plan Step 3: input = question + mark scheme + student steps,
output = per-step classification + justification. One training pair per
gold example, matching baseline.py's batched-per-question inference shape
(the model should learn to classify a whole submission in one pass, using
the rest of the working as context, same as the Option C baseline it's
meant to eventually replace).

This does NOT decide whether you have enough data to fine-tune — the
design plan's threshold is >=1000 examples; below that, SFT overfits and
underperforms the Option C/B baselines it's supposed to beat. See
finetune/README.md for the training recipe once you're past that
threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

from classifier.baseline import SYSTEM_TEMPLATE
from classifier.schema import GoldExample, MarkScheme

MARK_SCHEME_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "mark_schemes"


def _load_mark_schemes() -> dict[str, MarkScheme]:
    schemes = {}
    for path in MARK_SCHEME_DIR.glob("*.json"):
        scheme = MarkScheme.model_validate_json(path.read_text())
        schemes[scheme.question_id] = scheme
    return schemes


def format_example(example: GoldExample, scheme: MarkScheme) -> dict:
    system_text = SYSTEM_TEMPLATE.format(
        question_text=scheme.question_text,
        mark_scheme_json=scheme.model_dump_json(indent=2),
    )
    user_text = "Student's working, in order:\n" + "\n".join(
        f"Step {s.step_index}: {s.content}" for s in example.steps
    )
    output = {"steps": [sc.model_dump(exclude={"source"}) for sc in example.expected]}
    return {"system": system_text, "input": user_text, "output": json.dumps(output)}


def format_file(gold_path: Path, output_path: Path) -> int:
    schemes = _load_mark_schemes()
    count = 0
    with gold_path.open() as infile, output_path.open("w") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            example = GoldExample.model_validate_json(line)
            scheme = schemes[example.question_id]
            outfile.write(json.dumps(format_example(example, scheme)) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert gold JSONL to SFT training-pair JSONL.")
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    n = format_file(args.gold, args.output)
    print(f"Wrote {n} SFT training pair(s) to {args.output}")
