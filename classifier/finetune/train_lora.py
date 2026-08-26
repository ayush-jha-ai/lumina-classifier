"""Option A: QLoRA fine-tune on the SFT data produced by format_sft_data.py.

NOT run or verified end-to-end in development — no GPU was available, and
there isn't remotely enough gold data yet (the design plan's own rule:
don't attempt this under ~1,000 labelled examples, see
classifier/finetune/README.md for why). This is the documented recipe as
real code, gated so it refuses to run on too little data, ready for
whenever pilot data clears that bar. The heavy ML imports (torch,
transformers, peft, trl, bitsandbytes) are deferred until after the data
check, so `--help` and the gate check work without installing them.

Sanity-check the trl/peft/transformers call shapes below against whatever
versions you actually install before trusting this on a real run —
these APIs move; the shapes here reflect the documented, stable pattern
at time of writing, not a verified execution.

Usage:
    pip install -r classifier/finetune/requirements-finetune.txt
    python -m classifier.finetune.train_lora \
        --train data/sft/train.jsonl \
        --output-dir models/lumina-lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MIN_EXAMPLES = 1000


def load_sft_dataset(path: Path):
    from datasets import Dataset

    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        records.append(
            {
                "messages": [
                    {"role": "system", "content": row["system"]},
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]},
                ]
            }
        )
    return Dataset.from_list(records)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune a small open model on labelled classifier data.")
    parser.add_argument("--train", required=True, type=Path, help="SFT JSONL from format_sft_data.py")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Skip the {MIN_EXAMPLES}-example minimum (smoke-testing the pipeline only — "
        "accuracy from a run this small is meaningless).",
    )
    return parser


def count_examples(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def main() -> None:
    args = build_arg_parser().parse_args()

    n = count_examples(args.train)
    if n < MIN_EXAMPLES and not args.force:
        raise SystemExit(
            f"{args.train} has {n} example(s). The design plan's threshold is "
            f"~{MIN_EXAMPLES}+ before fine-tuning is worth attempting — below that you "
            "overfit and land worse than the hybrid rule+LLM baseline. Keep growing "
            "data/gold/ via classifier/distill.py + human correction first, or pass "
            "--force to smoke-test the pipeline anyway (not a real training run)."
        )

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_sft_dataset(args.train)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=4096,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    print(f"Saved LoRA adapter to {args.output_dir}")
    print(
        "Next: point classifier/eval/run_eval.py at this model's inference endpoint "
        "and confirm it beats --engine hybrid on a held-out gold set before it ships."
    )


if __name__ == "__main__":
    main()
