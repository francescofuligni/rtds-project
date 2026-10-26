"""Held-out income predictions and majority voting for selected tree prefixes."""

import numpy as np

UTILITY_PROTOCOL = {
    "utility_metric": "majority_vote",
    "tie_policy": "smallest_class",
    "evaluation": "disjoint_seeded_holdout",
    "preprocessing": "full_cleaned_table",
}


def predict_tree(tree: dict, X: np.ndarray) -> np.ndarray:
    """Route binary features using TreeFARMS true=1 / false=0 semantics."""
    if "prediction" in tree:
        prediction = tree["prediction"]
        if prediction not in (0, 1):
            raise ValueError("Expected binary income predictions")
        return np.full(len(X), prediction, dtype=np.int8)
    mask = X[:, tree["feature"]] == 1
    result = np.empty(len(X), dtype=np.int8)
    result[mask] = predict_tree(tree["true"], X[mask])
    result[~mask] = predict_tree(tree["false"], X[~mask])
    return result


def prefix_accuracy(trees: list[dict], X, y, m_values) -> dict[int, float]:
    """Evaluate nested prefixes; a tied binary vote predicts the smaller class."""
    X, y = np.asarray(X), np.asarray(y)
    requested = set(m_values)
    if not requested or min(requested) < 1 or max(requested) > len(trees):
        raise ValueError("Invalid release sizes")
    if len(X) == 0 or len(X) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected nonempty aligned binary holdout labels")
    votes = np.zeros(len(X), dtype=np.int64)
    scores = {}
    for m, tree in enumerate(trees, 1):
        votes += predict_tree(tree, X)
        if m in requested:
            scores[m] = float(((2 * votes > m) == y).mean())
    return scores
