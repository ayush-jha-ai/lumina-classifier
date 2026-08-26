import os
from pathlib import Path

import pytest

from classifier.distill import distill_file

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="requires ANTHROPIC_API_KEY"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_distill_writes_one_candidate_record_per_input_line(tmp_path):
    output = tmp_path / "candidates.jsonl"
    n = distill_file(DATA_DIR / "unlabelled" / "sample_raw_working.jsonl", output)
    assert n == 2
    lines = output.read_text().strip().splitlines()
    assert len(lines) == 2
