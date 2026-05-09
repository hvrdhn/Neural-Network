import random
import numpy as np
from torchvision import transforms, datasets

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases  = [np.random.randn(y, 1) * 0.1 for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0 / x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # ── activations ──────────────────────────────────────────────────────────
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_prime(z):
        return (z > 0).astype(float)   

    @staticmethod
    def softmax(z):
        z = z - z.max(axis=0, keepdims=True) # numerical stability
        e = np.exp(z)
        return e / e.sum(axis=0, keepdims=True)

    # ── inference (batched) ───────────────────────────────────────────────────
    def feedforward(self, a):
        """a: (784, batch) — works for batch=1 too."""
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            a = self.relu(w @ a + b)
        return self.softmax(self.weights[-1] @ a + self.biases[-1])

    # ── training ─────────────────────────────────────────────────────────────
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        # Pre-stack into big arrays once — avoids repeated reshaping
        X_train = np.hstack([x.reshape(784, 1) for x, _ in training_data])
        Y_train = np.hstack([y.reshape(10, 1)  for _, y in training_data])
        n = X_train.shape[1]

        if test_data:
            X_test = np.hstack([x.reshape(784, 1) for x, _ in test_data])
            Y_test = np.hstack([y.reshape(10, 1)  for _, y in test_data])

        for j in range(epochs):
            # Shuffle columns together
            perm = np.random.permutation(n)
            X_train, Y_train = X_train[:, perm], Y_train[:, perm]

            for k in range(0, n, mini_batch_size):
                xb = X_train[:, k:k + mini_batch_size]
                yb = Y_train[:, k:k + mini_batch_size]
                self._update_batch(xb, yb, eta)

            if test_data:
                acc = self._evaluate(X_test, Y_test)
                total = X_test.shape[1]
                print(f'Epoch {j}: {acc}/{total} ({acc/total*100:.2f}%)')
            else:
                print(f'Epoch {j} complete')

    def _update_batch(self, X, Y, eta):
        """Fully vectorized: X is (784, batch), Y is (10, batch)."""
        batch_size = X.shape[1]
        nabla_b, nabla_w = self._backprop_batch(X, Y)

        # In-place update — avoids creating new weight arrays
        for w, nw in zip(self.weights, nabla_w):
            w -= (eta / batch_size) * nw
        for b, nb in zip(self.biases, nabla_b):
            b -= (eta / batch_size) * nb

    def _backprop_batch(self, X, Y):
        """
        Vectorized backprop over a whole mini-batch simultaneously.
        Complexity: O(n_l * n_{l-1}) per layer — same big-O but
        BLAS matrix ops replace Python loops → ~10-30x faster in practice.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # ── forward ──
        activation = X
        activations = [X]
        zs = []

        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = w @ activation + b       # b broadcasts over batch dim
            zs.append(z)
            activation = self.relu(z)
            activations.append(activation)

        z_final = self.weights[-1] @ activation + self.biases[-1]
        zs.append(z_final)
        activations.append(self.softmax(z_final))

        # ── backward ──
        # delta shape: (n_out, batch) — gradients for all samples at once
        delta = activations[-1] - Y                          # cross-entropy + softmax gradient
        nabla_b[-1] = delta.sum(axis=1, keepdims=True)       # sum over batch
        nabla_w[-1] = delta @ activations[-2].T

        for l in range(2, self.num_layers):
            delta = (self.weights[-l+1].T @ delta) * self.relu_prime(zs[-l])
            nabla_b[-l] = delta.sum(axis=1, keepdims=True)
            nabla_w[-l] = delta @ activations[-l-1].T

        return nabla_b, nabla_w

    def _evaluate(self, X, Y):
        """Batched evaluation — one matrix multiply, not 10k loops."""
        preds  = np.argmax(self.feedforward(X), axis=0)
        labels = np.argmax(Y, axis=0)
        return int((preds == labels).sum())


# ── data loading (unchanged) ──────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_images = train_dataset.data.numpy().reshape(-1, 784, 1) / 255.0
test_images  = test_dataset.data.numpy().reshape(-1, 784, 1) / 255.0

def one_hot(y, n=10):
    e = np.zeros(n); e[y] = 1.0; return e

train_labels = [one_hot(y) for y in train_dataset.targets.numpy()]
test_labels  = [one_hot(y) for y in test_dataset.targets.numpy()]

training_data = list(zip(train_images, train_labels))
test_data     = list(zip(test_images,  test_labels))

net = Network([784, 128, 64, 10])
net.SGD(training_data, epochs=20, eta=0.01, mini_batch_size=32, test_data=test_data) 

import matplotlib
import matplotlib.pyplot as plt

# Pick a random test sample
index = np.random.randint(len(test_data))
test_image, test_label = test_data[index]

# Predict
prediction = net.feedforward(test_image)
predicted_digit = np.argmax(prediction)

# Show image
plt.imshow(test_image.reshape(28, 28), cmap="gray")
plt.title(f"Predicted: {predicted_digit}, Actual: {np.argmax(test_label)}")
plt.show()
