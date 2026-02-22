"""
Gaussian Naive Bayes from Scratch
==================================
Assumes features follow a Gaussian distribution within each class.

P(y|X) = P(X|y) * P(y) / P(X)
P(xi|y) = (1 / sqrt(2*pi*var)) * exp(-(xi - mean)^2 / (2*var))
"""

import numpy as np


class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-9  # smoothing
            self.priors[c] = X_c.shape[0] / X.shape[0]
        return self

    def _gaussian_likelihood(self, x, mean, var):
        coeff = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coeff * exponent

    def _predict_single(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            likelihood = np.sum(np.log(self._gaussian_likelihood(x, self.mean[c], self.var[c])))
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=300, n_features=4, n_redundant=0, random_state=42)
    model = GaussianNaiveBayes()
    model.fit(X, y)
    acc = (model.predict(X) == y).mean()
    print(f"Gaussian Naive Bayes Accuracy: {acc:.2%}")
