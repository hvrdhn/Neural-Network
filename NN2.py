import random
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms, datasets
import torch

class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) * 0.1 for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0/x) for x, y in zip(sizes[:-1], sizes[1:])]

    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        z = z - np.max(z, axis=0, keepdims=True)
        e_z = np.exp(z)
        return e_z / np.sum(e_z, axis=0, keepdims=True)

    def feedforward(self, a):
        a = np.array(a, dtype=np.float64).reshape(-1, 1)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, a) + b
            a = self.relu(z)
        return self.softmax(np.dot(self.weights[-1], a) + self.biases[-1])
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = [(np.reshape(x, (784, 1)), np.reshape(y, (10, 1))) for x, y in training_data]
        if test_data:
            test_data = [(np.reshape(x, (784, 1)), np.reshape(y, (10, 1))) for x, y in test_data]
            
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                accuracy = self.evaluate(test_data)
                print(f'Epoch {j}: {accuracy}/{len(test_data)} ({(accuracy/len(test_data))*100:.2f}%)')
            else:
                print(f'Epoch {j} complete')

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        # Forward pass
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.relu(z)
            activations.append(activation)
        
        # Output layer
        z_final = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z_final)
        activation = self.softmax(z_final)
        activations.append(activation)

        # Backward pass
        delta = self.cost_derivate(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.relu_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
            
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def cost_derivate(self, output_activations, y):
        return output_activations - y

    @staticmethod
    def relu_prime(z):
        return (z > 0).astype(float)
    



# Define a transformation (convert images to tensors & normalize)
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize using MNIST mean & std
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Create DataLoader to batch and shuffle data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Check dataset size
print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")




# Convert images & labels to NumPy format
train_images = train_dataset.data.numpy().reshape(-1, 28*28, 1) / 255.0  # Normalize
train_labels = train_dataset.targets.numpy()

test_images = test_dataset.data.numpy().reshape(-1, 28*28, 1) / 255.0
test_labels = test_dataset.targets.numpy()

# One-hot encode labels
def one_hot_encode(y, num_classes=10):
    encoded = np.zeros((num_classes, ))
    encoded[y] = 1.0
    return encoded

train_labels = [one_hot_encode(y) for y in train_labels]
test_labels = [one_hot_encode(y) for y in test_labels]

# Create dataset in your required format
training_data = list(zip(train_images, train_labels))
test_data = list(zip(test_images, test_labels))

print(f"Converted training samples: {len(training_data)}, Test samples: {len(test_data)}")

# Initialize your network (784 input, 30 hidden, 10 output)
net = Network([784, 128, 64, 10])

# Train with stochastic gradient descent
net.SGD(training_data, epochs=40, eta=0.01, mini_batch_size=32, test_data=test_data)


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

