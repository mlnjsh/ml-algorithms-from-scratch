"""
Support Vector Machine from Scratch
=====================================
Linear SVM classifier using hinge loss and subgradient descent.

Loss: L = (1/n) * sum(max(0, 1 - y_i * (w . x_i + b))) + lambda * ||w||^2
"""

import numpy as np


class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Convert labels to -1, 1
        y_ = np.where(y <= 0, -1, 1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx in range(n_samples):
                condition = y_[idx] * (X[idx] @ self.weights + self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (
                        2 * self.lambda_param * self.weights - y_[idx] * X[idx]
                    )
                    self.bias -= self.lr * (-y_[idx])

            # Compute hinge loss
            margins = y_ * (X @ self.weights + self.bias)
            hinge = np.maximum(0, 1 - margins)
            loss = np.mean(hinge) + self.lambda_param * np.sum(self.weights ** 2)
            self.loss_history.append(loss)

        return self

    def predict(self, X):
        linear_output = X @ self.weights + self.bias
        return np.where(np.sign(linear_output) == -1, 0, 1)


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, random_state=42)
    model = SVM(lr=0.001, lambda_param=0.01, n_iters=500)
    model.fit(X, y)
    acc = (model.predict(X) == y).mean()
    print(f"SVM Accuracy: {acc:.2%}")
