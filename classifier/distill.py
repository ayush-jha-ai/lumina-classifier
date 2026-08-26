"""Design plan Step 1: seed dataset via distillation.

Runs the Option C baseline classifier over unlabelled raw submissions to
produce first-pass candidate labels for human correction. This is
explicitly a starting point, not ground truth — every record is written
with reviewed: false and is expected to be hand-corrected (by a teacher, or
the founder for the GCSE maths beachhead) before it's promoted into
data/gold/.

Input: JSONL, one record per line:
  {"question_id": "...", "student_id": "...", "steps": [{"step_index": 1, "content": "..."}]}

Output: JSONL, one record per line:
  {"question_id": "...", "student_id": "...", "steps": [...classified...], "reviewed": false}
"""

from __future__ import annotations

import json
from pathlib import Path

from classifier.baseline import classify
from classifier.schema import MarkScheme, StudentStep

MARK_SCHEME_DIR = Path(__file__).resolve().parent.parent / "data" / "mark_schemes"


def _load_mark_scheme(question_id: str) -> MarkScheme:
    for path in MARK_SCHEME_DIR.glob("*.json"):
        scheme = MarkScheme.model_validate_json(path.read_text())
        if scheme.question_id == question_id:
            return scheme
    raise FileNotFoundError(f'No mark scheme found for question_id "{question_id}" under {MARK_SCHEME_DIR}')


def distill_file(input_path: Path, output_path: Path) -> int:
    """Distill every record in `input_path` and append results to `output_path`. Returns record count."""
    count = 0
    with input_path.open() as infile, output_path.open("a") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            scheme = _load_mark_scheme(record["question_id"])
            steps = [{"step_index": s["step_index"], "content": s["content"]} for s in record["steps"]]
            result = classify(scheme, record["student_id"], steps)
            payload = result.model_dump()
            payload["reviewed"] = False
            outfile.write(json.dumps(payload) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Distill candidate labels for unlabelled submissions.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "distilled" / "candidates.jsonl",
    )
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set — see .env.example.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = distill_file(args.input, args.output)
    print(f"Wrote {n} candidate-labelled record(s) to {args.output}")
