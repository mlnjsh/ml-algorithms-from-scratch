"""
Logistic Regression from Scratch
================================
Binary classification using sigmoid function and cross-entropy loss.

Model: P(y=1|x) = sigmoid(Xw + b)
Loss:  L = -(1/n) * sum(y*log(p) + (1-y)*log(1-p))
"""

import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)

            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            eps = 1e-15
            loss = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, random_state=42)
    model = LogisticRegression(lr=0.1, n_iters=1000)
    model.fit(X, y)
    acc = (model.predict(X) == y).mean()
    print(f"Accuracy: {acc:.2%}")
    print(f"Final loss: {model.loss_history[-1]:.4f}")
