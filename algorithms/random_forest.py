"""
Random Forest Classifier from Scratch
======================================
Ensemble of decision trees using bootstrap aggregation and feature subsampling.

1. Draw bootstrap sample from training data
2. At each split, consider random subset of features
3. Aggregate predictions via majority vote
"""

import numpy as np
from collections import Counter
from .decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, max_features="sqrt"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            X_boot, y_boot = self._bootstrap_sample(X, y)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Majority vote across trees for each sample
        final_preds = []
        for i in range(X.shape[0]):
            votes = tree_preds[:, i]
            final_preds.append(Counter(votes).most_common(1)[0][0])
        return np.array(final_preds)


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=300, n_features=4, n_redundant=0, random_state=42)
    model = RandomForest(n_trees=10, max_depth=5)
    model.fit(X, y)
    acc = (model.predict(X) == y).mean()
    print(f"Random Forest (10 trees) Accuracy: {acc:.2%}")
