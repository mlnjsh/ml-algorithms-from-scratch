"""
Linear Regression from Scratch
==============================
Supports both gradient descent and closed-form (normal equation) solutions.

Model: y = Xw + b
Loss:  L = (1/2n) * sum((y_pred - y)^2)
Update: w -= lr * (1/n) * X^T (Xw - y)
"""

import numpy as np


class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000, method="gradient_descent"):
        self.lr = lr
        self.n_iters = n_iters
        self.method = method
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        if self.method == "normal_equation":
            # Closed-form: w = (X^T X)^(-1) X^T y
            X_b = np.c_[np.ones((n_samples, 1)), X]
            theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            self.bias = theta[0]
            self.weights = theta[1:]
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0

            for _ in range(self.n_iters):
                y_pred = X @ self.weights + self.bias
                error = y_pred - y

                dw = (1 / n_samples) * (X.T @ error)
                db = (1 / n_samples) * np.sum(error)

                self.weights -= self.lr * dw
                self.bias -= self.lr * db

                loss = np.mean(error ** 2) / 2
                self.loss_history.append(loss)

        return self

    def predict(self, X):
        return X @ self.weights + self.bias

    def r2_score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


if __name__ == "__main__":
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=200, n_features=1, noise=15, random_state=42)

    model_gd = LinearRegression(lr=0.01, n_iters=1000, method="gradient_descent")
    model_gd.fit(X, y)
    print(f"[Gradient Descent] R2 = {model_gd.r2_score(X, y):.4f}")

    model_ne = LinearRegression(method="normal_equation")
    model_ne.fit(X, y)
    print(f"[Normal Equation]  R2 = {model_ne.r2_score(X, y):.4f}")
