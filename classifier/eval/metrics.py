"""Design plan Step 4: evaluation harness metrics.

Accuracy alone isn't enough — "correct" is the majority class, and
knowledge_gap is the rarer, more important one to catch (per the design
doc). This reports accuracy plus a full confusion matrix and per-class
precision/recall derived from it.

Deliberately not included yet: Cohen's kappa (human-human agreement
ceiling) and topic-stratified breakdowns. Both need a second human labeler
and a gold set much larger than the ~15-20 seed examples this repo ships
with — add them once pilot data and a second labeler exist (design doc
Step 4).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    n: int
    accuracy: float
    confusion_matrix: dict[str, dict[str, int]]
    per_class: dict[str, dict[str, float | int | None]]


def compute_metrics(pairs: list[tuple[str, str]]) -> Metrics:
    """`pairs` is a list of (gold_label, predicted_label)."""
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for gold, pred in pairs:
        confusion[gold][pred] += 1

    n = len(pairs)
    correct = sum(confusion[label][label] for label in labels)
    accuracy = correct / n if n else 0.0

    per_class: dict[str, dict[str, float | int | None]] = {}
    for label in labels:
        tp = confusion[label][label]
        fn = sum(confusion[label][pred] for pred in labels if pred != label)
        fp = sum(confusion[gold][label] for gold in labels if gold != label)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        per_class[label] = {"precision": precision, "recall": recall, "support": tp + fn}

    return Metrics(n=n, accuracy=accuracy, confusion_matrix=confusion, per_class=per_class)


def format_report(metrics: Metrics) -> str:
    lines = [f"n={metrics.n}  accuracy={metrics.accuracy:.3f}", "", "per-class precision/recall/support:"]
    for label, stats in metrics.per_class.items():
        precision = f"{stats['precision']:.3f}" if stats["precision"] is not None else "  n/a"
        recall = f"{stats['recall']:.3f}" if stats["recall"] is not None else "  n/a"
        lines.append(f"  {label:<18} precision={precision}  recall={recall}  support={stats['support']}")

    labels = list(metrics.confusion_matrix.keys())
    lines.append("")
    lines.append("confusion matrix (rows=gold, cols=predicted):")
    header = " " * 18 + "".join(f"{label[:12]:>14}" for label in labels)
    lines.append(header)
    for gold in labels:
        row = "".join(f"{metrics.confusion_matrix[gold][pred]:>14}" for pred in labels)
        lines.append(f"  {gold:<16}{row}")

    return "\n".join(lines)
