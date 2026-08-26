# Lumina diagnostic classifier

Classifies each step of a GCSE maths student's working, against a mark
scheme (M1/A1/B1 criteria), as `correct`, `process_error` (right method,
execution slip), or `knowledge_gap` (wrong method). Built per
`lumina-classifier-design-plan.md`.

This is **not** `lumina-live-mvp` — that's a separate, already-built
Next.js demo app with a deterministic, LLM-free step matcher used for
investor/school demos. This repo is the real diagnostic engine, meant to
eventually power the actual pilot product.

## Sequencing (per the design plan)

| Option | What | Status |
|---|---|---|
| **C** | Prompted LLM baseline (`classifier/baseline.py`) | Built, needs `ANTHROPIC_API_KEY` to run |
| **B** | Rule layer + targeted LLM fallback (`classifier/extraction.py`) | Built — the version meant to ship for the pilot |
| **A** | LoRA/QLoRA fine-tune on a small open model | Not built — see `classifier/finetune/README.md`. Design plan's own rule: don't attempt this under ~1,000 labelled examples |

Never ship a change to Option B/A to a live pilot without it beating the
current baseline on `classifier/eval/run_eval.py`'s gold set first.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY
```

## Run it

```bash
# Unit tests — extraction.py and metrics.py run with no API key;
# baseline.py/distill.py tests skip cleanly without one.
pytest

# Rule-only eval (deterministic, no API key needed)
python -m classifier.eval.run_eval --gold data/gold/gold_examples.jsonl --engine rule

# Hybrid (Option B) and LLM-only (Option C) eval — needs ANTHROPIC_API_KEY
python -m classifier.eval.run_eval --gold data/gold/gold_examples.jsonl --engine hybrid
python -m classifier.eval.run_eval --gold data/gold/gold_examples.jsonl --engine llm

# Distill candidate labels for unlabelled submissions (design plan Step 1)
python -m classifier.distill --input data/unlabelled/sample_raw_working.jsonl
```

`--engine rule` intentionally can't resolve every case — any step that
doesn't match a mark scheme's `expected_forms` or `error_patterns`
verbatim falls back to a hardcoded `knowledge_gap` label rather than
calling the LLM. The gold set includes a couple of examples designed to
expose this (`sim-D-unenumerated-slip`, `quad-D-unenumerated-slip` — real
process errors that aren't in the hand-authored error-pattern list) — they
show up as mismatches under `--engine rule` and are exactly what
`--engine hybrid` exists to fix.

## Rate limiting

Every call to the Anthropic API — from `baseline.py` directly, from
`extraction.py`'s LLM fallback, or from `distill.py` — is throttled by a
shared client-side limiter (`classifier/rate_limit.py`), so a naive loop
over many examples in `run_eval.py`/`distill.py` can't hammer the API.
Default: 20 calls/minute. Override with `LUMINA_RATE_LIMIT_CALLS_PER_MINUTE`
in `.env`. This is in addition to, not instead of, the SDK's own automatic
retry-with-backoff on 429s.

## Layout

- `classifier/schema.py` — shared Pydantic models (`MarkScheme`, `StudentStep`, `StepClassification`, `GoldExample`, ...).
- `classifier/baseline.py` — Option C.
- `classifier/extraction.py` — Option B; ports the exact-match logic from `lumina-live-mvp/src/lib/diagnosis.ts`, routing anything unresolved to `baseline.py` for the process-error-vs-knowledge-gap call.
- `classifier/distill.py` — design plan Step 1 (seed-dataset generation via distillation).
- `classifier/eval/` — design plan Step 4 (accuracy, confusion matrix, per-class precision/recall). Cohen's kappa and topic-stratified breakdowns are deliberately not built yet — they need a second human labeler and a much larger gold set than the ~16-step seed set here provides.
- `classifier/finetune/` — Option A recipe and SFT data formatter (not run yet).
- `data/mark_schemes/` — 2 hand-authored generalized mark schemes (simultaneous equations, quadratic factorising), matching the design plan's own recommended starting topics.
- `data/gold/gold_examples.jsonl` — hand-written gold set.
- `data/unlabelled/`, `data/distilled/` — inputs/outputs for `distill.py`.

## Growing the gold set

Real pilot data is the actual moat here, per the design plan. As
submissions come in: drop raw (unlabelled) ones into `data/unlabelled/`,
run `classifier/distill.py` to get first-pass candidate labels, hand-correct
them, and move the corrected records into `data/gold/`. Every correction
is signal — that's the ambiguous process-vs-knowledge-gap boundary the
design plan flags as the actual hard part of this project.
