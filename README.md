# Lumina Classifier

Given a GCSE maths question, its official mark scheme, and a student's
step-by-step working, this classifies **each step** as one of:

- **`correct`** — satisfies the relevant mark-scheme criterion
- **`process_error`** — right method, procedural slip (arithmetic error, sign error, mis-substitution)
- **`knowledge_gap`** — the method itself is wrong; the student doesn't yet understand what this step requires

That distinction — slip vs. gap — is the actual product. "Wrong" isn't a
useful signal to a teacher on its own; *why* it's wrong determines whether
the fix is "practice this arithmetic" or "re-teach this concept."

## Why this is a hard classification problem, not a trivial one

Correct-vs-incorrect is easy: check the step against the mark scheme. The
hard part is that "incorrect" splits into two categories that look similar
in isolation. A wrong substitution can be a slip (the student knew the
formula, mistyped a number) or a gap (the student didn't understand what
to substitute) — and disambiguating the two often requires looking at the
*rest* of the student's working for consistency, not just the failing step.
This repo's design leans directly into that: the rule layer resolves the
easy cases deterministically and cheaply, and only the genuinely ambiguous
cases go to an LLM that gets the full submission as context.

## How it works — three tiers, in increasing order of sophistication

| Tier | Approach | File | Status |
|---|---|---|---|
| **Baseline** | A single prompted call to Claude classifies every step of a submission at once, so the model can use the rest of the working to resolve ambiguity. | [`classifier/baseline.py`](classifier/baseline.py) | Working, needs an API key |
| **Hybrid** *(production path)* | A deterministic rule layer (exact-match against the mark scheme's expected forms and a library of known error patterns) resolves what it can for free and with full interpretability; anything it can't resolve is batched and handed to the baseline model. | [`classifier/extraction.py`](classifier/extraction.py) | Working — this is the version meant to ship |
| **Fine-tuned** | A small open model (e.g. Qwen2.5-7B) fine-tuned on real labelled submissions once there's enough of them (~1,000+). | [`classifier/finetune/`](classifier/finetune) | Recipe documented, not yet trained — no fine-tune ships until it beats the hybrid tier on held-out data |

The hybrid tier is deliberately not "just call an LLM on everything": every
rule-resolved classification traces back to an explicit, auditable rule a
teacher (or an investor doing diligence) can read — the LLM is reserved for
the cases that genuinely need judgment.

## Quickstart

```bash
git clone https://github.com/ayush-jha-ai/lumina-classifier.git
cd lumina-classifier

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# open .env and set ANTHROPIC_API_KEY — required for the hybrid/baseline
# tiers; the rule-only tier and all tests run without one.
```

## See it work: a worked example

`data/gold/gold_examples.jsonl` ships with hand-labelled example
submissions. Here's one — a student solving simultaneous equations who
makes an arithmetic slip, then self-corrects:

```
Question: Solve the simultaneous equations: 2x + y = 10 and x - y = 2

Step 1: 3x = 8    <- should be 3x = 12 (10 + 2 added incorrectly)
Step 2: 3x = 12
Step 3: x = 4
Step 4: y = 2
```

Running the hybrid classifier on this produces:

```json
{
  "steps": [
    { "step_index": 1, "content": "3x=8",  "label": "process_error",
      "criterion": "M1", "source": "rule",
      "justification": "Added the right-hand sides incorrectly (10 + 2 miscalculated as 8)." },
    { "step_index": 2, "content": "3x=12", "label": "correct",
      "criterion": "M1", "source": "rule",
      "justification": "Matches M1: Add the two equations to eliminate y." },
    { "step_index": 3, "content": "x=4",   "label": "correct",
      "criterion": "A1", "source": "rule",
      "justification": "Matches A1: Divide both sides by 3 to find x." },
    { "step_index": 4, "content": "y=2",   "label": "correct",
      "criterion": "A1", "source": "rule",
      "justification": "Matches A1: Substitute x = 4 into one original equation to find y." }
  ]
}
```

Every field here is resolved by the deterministic rule layer (`source:
"rule"`) — no API call needed. `source` flips to `"llm"` only for steps the
rule layer couldn't place, so you can always see which classifications are
guaranteed-traceable and which came from the model's judgment.

## Usage

### As a script

```bash
# Rule-only — deterministic, free, no API key
python -m classifier.eval.run_eval --gold data/gold/gold_examples.jsonl --engine rule

# Hybrid — the production tier (needs ANTHROPIC_API_KEY)
python -m classifier.eval.run_eval --gold data/gold/gold_examples.jsonl --engine hybrid

# Baseline-only, for comparison (needs ANTHROPIC_API_KEY)
python -m classifier.eval.run_eval --gold data/gold/gold_examples.jsonl --engine llm

# Generate first-pass candidate labels for new, unlabelled submissions
python -m classifier.distill --input data/unlabelled/sample_raw_working.jsonl
```

### As a library

```python
from classifier.extraction import classify
from classifier.schema import MarkScheme, StudentStep

scheme = MarkScheme.model_validate_json(
    open("data/mark_schemes/simultaneous_equations.json").read()
)
steps = [
    StudentStep(step_index=1, content="3x=12"),
    StudentStep(step_index=2, content="x=4"),
    StudentStep(step_index=3, content="y=2"),
]
result = classify(scheme, student_id="demo-student", steps=steps)
for step in result.steps:
    print(step.step_index, step.label, step.justification)
```

## Evaluation

Every step of every gold example gets a prediction, compared label-for-label
against the hand-written ground truth. Running the rule-only tier against
the seed gold set today:

```
n=16  accuracy=0.875

per-class precision/recall/support:
  correct            precision=1.000  recall=1.000  support=10
  knowledge_gap      precision=0.500  recall=1.000  support=2
  process_error      precision=1.000  recall=0.500  support=4
```

The two misses are intentional: the gold set includes process errors that
aren't in the hand-authored error-pattern library, specifically to
demonstrate what the rule-only tier can't catch on its own — that gap is
exactly what the hybrid tier's LLM fallback exists to close. Nothing here
is cherry-picked to look good; run it yourself and read the mismatches.

No fine-tuned model, and no change to the hybrid tier's rule layer, is
meant to ship without first beating this baseline on a held-out set —
that gate is the point of `classifier/eval/`, not an afterthought.

## Project layout

```
classifier/
  schema.py          shared data model (MarkScheme, StepClassification, ...)
  baseline.py         prompted-LLM classifier
  extraction.py        rule layer + LLM fallback (the production tier)
  distill.py            generates first-pass labels for unlabelled data
  rate_limit.py          throttles outgoing API calls (see below)
  eval/                  accuracy / confusion matrix / precision-recall harness
  finetune/               fine-tuning recipe + SFT data formatter
data/
  mark_schemes/       hand-authored mark schemes (JSON)
  gold/                hand-labelled ground truth
  unlabelled/, distilled/  input/output for distill.py
tests/                 pytest suite (LLM-dependent tests skip without a key)
```

## Rate limiting

Every outbound call to the Anthropic API — whichever tier triggers it —
passes through a single shared, thread-safe throttle
(`classifier/rate_limit.py`), so running the eval harness or the
distillation script over a batch of examples can't hammer the API or run
up unexpected cost. Default: 20 calls/minute, overridable via
`LUMINA_RATE_LIMIT_CALLS_PER_MINUTE` in `.env`.

## Where the real data moat comes from

Hand-authored mark schemes and gold examples get this off the ground, but
the two seed topics here (simultaneous equations, quadratic factorising)
are a start, not the product. As real student submissions come in: drop
the raw, unlabelled ones into `data/unlabelled/`, run `classifier/distill.py`
to get first-pass candidate labels, hand-correct them, and promote the
corrected records into `data/gold/`. Every correction is signal on exactly
the process-error-vs-knowledge-gap boundary that's the hard part of this
problem — that accumulated, corrected dataset is what eventually makes the
fine-tuned tier worth training.
