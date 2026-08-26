"""Covers only the parts of train_lora.py that don't need torch/transformers/
peft/trl installed: CLI parsing and the data-volume gate. The actual
training path is unverified in this environment — see the module docstring.
"""

import pytest

from classifier.finetune.train_lora import MIN_EXAMPLES, build_arg_parser, count_examples, main


def test_defaults():
    args = build_arg_parser().parse_args(["--train", "x.jsonl", "--output-dir", "out/"])
    assert args.base_model == "Qwen/Qwen2.5-7B-Instruct"
    assert args.epochs == 3.0
    assert args.lora_r == 16
    assert args.force is False


def test_count_examples_skips_blank_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n  \n{"a": 3}\n')
    assert count_examples(path) == 3


def test_main_refuses_to_train_below_threshold(tmp_path, monkeypatch):
    train_path = tmp_path / "train.jsonl"
    train_path.write_text('{"system": "s", "input": "i", "output": "o"}\n' * (MIN_EXAMPLES - 1))
    monkeypatch.setattr(
        "sys.argv",
        ["train_lora.py", "--train", str(train_path), "--output-dir", str(tmp_path / "out")],
    )
    with pytest.raises(SystemExit, match="threshold"):
        main()
