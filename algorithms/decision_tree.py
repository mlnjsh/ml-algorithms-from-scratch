"""
Decision Tree Classifier from Scratch
======================================
Uses recursive binary splitting with information gain (entropy) or Gini impurity.

Split criterion: argmax feature,threshold [ IG(parent) - weighted_avg(IG(children)) ]
"""

import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf node class label

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, criterion="entropy"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.n_features = None

    def _entropy(self, y):
        counts = np.bincount(y)
        probs = counts[counts > 0] / len(y)
        return -np.sum(probs * np.log2(probs))

    def _gini(self, y):
        counts = np.bincount(y)
        probs = counts[counts > 0] / len(y)
        return 1 - np.sum(probs ** 2)

    def _impurity(self, y):
        if self.criterion == "gini":
            return self._gini(y)
        return self._entropy(y)

    def _information_gain(self, parent, left_child, right_child):
        n = len(parent)
        n_left, n_right = len(left_child), len(right_child)
        parent_imp = self._impurity(parent)
        child_imp = (n_left / n) * self._impurity(left_child) + \
                    (n_right / n) * self._impurity(right_child)
        return parent_imp - child_imp

    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature in range(self.n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gain = self._information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or len(y) < self.min_samples_split:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        left_mask = X[:, feature] <= threshold
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y)
        return self

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=4, n_redundant=0, random_state=42)
    model = DecisionTree(max_depth=5)
    model.fit(X, y)
    acc = (model.predict(X) == y).mean()
    print(f"Decision Tree Accuracy: {acc:.2%}")
