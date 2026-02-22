"""
Neural Network from Scratch
============================
Feedforward neural network with backpropagation.
Supports configurable hidden layers, ReLU/sigmoid activations, and softmax output.

Architecture: Input -> [Hidden(ReLU)] * n -> Output(Softmax)
Loss: Cross-entropy
"""

import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.01, n_iters=1000):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.n_iters = n_iters
        self.weights = []
        self.biases = []
        self.loss_history = []

    def _init_weights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            # He initialization
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(
                2.0 / self.layer_sizes[i]
            )
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def _softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _forward(self, X):
        activations = [X]
        pre_activations = []

        for i in range(len(self.weights) - 1):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_activations.append(z)
            activations.append(self._relu(z))

        # Output layer with softmax
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        pre_activations.append(z)
        activations.append(self._softmax(z))

        return activations, pre_activations

    def _cross_entropy_loss(self, y_pred, y_true):
        n = y_true.shape[0]
        eps = 1e-15
        return -np.sum(y_true * np.log(y_pred + eps)) / n

    def fit(self, X, y):
        n_classes = len(np.unique(y))
        # One-hot encode
        y_onehot = np.zeros((len(y), n_classes))
        y_onehot[np.arange(len(y)), y] = 1

        self.layer_sizes = [X.shape[1]] + self.layer_sizes + [n_classes]
        self._init_weights()

        for epoch in range(self.n_iters):
            activations, pre_activations = self._forward(X)

            loss = self._cross_entropy_loss(activations[-1], y_onehot)
            self.loss_history.append(loss)

            # Backpropagation
            n = X.shape[0]
            delta = (activations[-1] - y_onehot) / n

            for i in range(len(self.weights) - 1, -1, -1):
                dw = activations[i].T @ delta
                db = np.sum(delta, axis=0, keepdims=True)

                if i > 0:
                    delta = (delta @ self.weights[i].T) * self._relu_derivative(pre_activations[i - 1])

                self.weights[i] -= self.lr * dw
                self.biases[i] -= self.lr * db

        return self

    def predict(self, X):
        activations, _ = self._forward(X)
        return np.argmax(activations[-1], axis=1)


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=500, n_features=10, n_classes=3,
                               n_informative=8, n_redundant=0, random_state=42)
    model = NeuralNetwork(layer_sizes=[32, 16], lr=0.1, n_iters=500)
    model.fit(X, y)
    acc = (model.predict(X) == y).mean()
    print(f"Neural Network Accuracy: {acc:.2%}")
    print(f"Final Loss: {model.loss_history[-1]:.4f}")
