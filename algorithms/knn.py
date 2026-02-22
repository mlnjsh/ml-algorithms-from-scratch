"""
K-Nearest Neighbors from Scratch
=================================
Non-parametric classifier using distance-based majority voting.

Prediction: y = mode of k nearest training labels
Distances:  Euclidean or Manhattan
"""

import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=5, metric="euclidean"):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self

    def _distance(self, x1, x2):
        if self.metric == "manhattan":
            return np.sum(np.abs(x1 - x2), axis=1)
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def _predict_single(self, x):
        distances = self._distance(self.X_train, x)
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        return Counter(k_labels).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=4, n_redundant=0, random_state=42)
    model = KNN(k=5)
    model.fit(X, y)
    acc = (model.predict(X) == y).mean()
    print(f"KNN (k=5) Training Accuracy: {acc:.2%}")
