from classifier.eval.metrics import compute_metrics


def test_accuracy_and_confusion_matrix():
    pairs = [
        ("correct", "correct"),
        ("correct", "correct"),
        ("process_error", "knowledge_gap"),
        ("knowledge_gap", "knowledge_gap"),
    ]
    m = compute_metrics(pairs)
    assert m.n == 4
    assert m.accuracy == 0.75
    assert m.confusion_matrix["process_error"]["knowledge_gap"] == 1
    assert m.per_class["correct"]["precision"] == 1.0
    assert m.per_class["correct"]["recall"] == 1.0
    assert m.per_class["process_error"]["recall"] == 0.0


def test_empty_pairs_returns_zero_accuracy_not_a_crash():
    m = compute_metrics([])
    assert m.n == 0
    assert m.accuracy == 0.0
    assert m.confusion_matrix == {}


def test_class_with_no_predictions_has_none_precision():
    pairs = [("correct", "correct"), ("knowledge_gap", "correct")]
    m = compute_metrics(pairs)
    # "knowledge_gap" was never predicted, so precision is undefined (0/0), not 0.0
    assert m.per_class["knowledge_gap"]["precision"] is None
