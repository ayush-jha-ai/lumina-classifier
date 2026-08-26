# Option A — fine-tuned open model (background R&D)

Not built this session — no GPU, and nowhere near the data volume the
design plan requires. This documents the recipe so it's a quick job once
pilot data justifies it.

## When to actually do this

Per the design plan: don't fine-tune under ~1,000 labelled (step,
criterion, label) examples — below that you'll overfit and land worse than
the Option C/B baselines. Feed pilot submissions into `classifier/distill.py`
→ human correction → `data/gold/` until you're past that threshold.

Never swap this in for the live pilot until it beats the current
production engine (Option B / `hybrid` in `run_eval.py`) on the held-out
eval set — that's the gate, not a vibe check.

## Recipe

1. **Data**: `python -m classifier.finetune.format_sft_data --gold data/gold/gold_examples.jsonl --output data/sft/train.jsonl`. Split off a held-out slice first — don't format the eval set into training data.
2. **Base model**: `Qwen2.5-7B-Instruct` (or `Llama-3.1-8B-Instruct`) — small enough for LoRA/QLoRA on a single consumer/cloud GPU, strong enough as an instruction-following base for a narrow classification task.
3. **Method**: QLoRA via `peft` + `trl`'s `SFTTrainer`. 4-bit quant (`bitsandbytes`), LoRA rank 16-32 on the attention/MLP projection layers, 2-3 epochs, small learning rate (~2e-4 with a LoRA-appropriate scheduler). This is a narrow, well-specified classification task per the design plan — plain SFT is right; don't reach for RLHF/DPO here.
4. **Format**: each training pair's `system` + `input` fields become the prompt, `output` (a JSON string: `{"steps": [...]}`) is the target completion — mirrors `baseline.py`'s actual inference shape, so the fine-tune is learning to reproduce the same interface it's meant to replace.
5. **Eval**: point `classifier/eval/run_eval.py` at the fine-tuned model's inference endpoint (add an `engine="finetuned"` branch once it exists) and compare its report against `hybrid` and `llm` on the same gold set, stratified by topic and label per the design plan's Step 4.

## Dependencies (not in the base `requirements.txt`)

```
transformers
peft
trl
bitsandbytes
datasets
accelerate
```
