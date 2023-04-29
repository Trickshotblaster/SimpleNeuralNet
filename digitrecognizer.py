import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json


def find_highest(arr):
    highest = 0
    index = 0
    for i, v in enumerate(arr):
        if v > highest:
            highest = v
            index = i
    return [index, round(highest, 3)]


class CustomNeuralNetwork:
    def __init__(self, layer_sizes, weights=[], biases=[], save_file="nn_weights_biases.txt"):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.activations = []
        self.save_file = save_file
        if not weights:
            # Initialize weights and biases
            for i in range(len(layer_sizes) - 1):
                self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))

        else:
            self.weights = weights

        if not biases:
            for i in range(len(layer_sizes) - 1):
                self.biases.append(np.zeros(layer_sizes[i + 1]))
        else:
            self.biases = biases

    def serialize(self):
        weights_serialized = [w.tolist() for w in self.weights]
        biases_serialized = [b.tolist() for b in self.biases]
        return json.dumps({"weights": weights_serialized, "biases": biases_serialized})

    def save_to_file(self, filename):
        serialized_data = self.serialize()
        with open(filename, "w") as f:
            f.write(serialized_data)

    def load_from_file(self, filename):
        with open(filename, "r") as f:
            serialized_data = f.read()
        self.deserialize(serialized_data)

    def deserialize(self, serialized_data):
        data = json.loads(serialized_data)
        weights = [np.array(w) for w in data["weights"]]
        biases = [np.array(b) for b in data["biases"]]
        layer_sizes = [w.shape[0] for w in weights] + [weights[-1].shape[1]]
        self.weights = weights
        self.biases = biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        layer_input = X
        self.activations = []

        for i in range(len(self.layer_sizes) - 1):
            layer_output = np.dot(layer_input, self.weights[i]) + self.biases[i]
            layer_input = self.sigmoid(layer_output)
            self.activations.append(layer_input)

        return layer_input

    def backpropagate(self, X, y, learning_rate):
        output = self.forward(X)
        deltas = []

        # Calculate output layer error
        error = output - y
        print(np.average(error))
        delta = error * self.sigmoid_derivative(output)
        deltas.append(delta)

        # Calculate hidden layer errors
        for i in reversed(range(len(self.weights) - 1)):
            error = np.dot(delta, self.weights[i + 1].T)
            delta = error * self.sigmoid_derivative(self.activations[i])
            deltas.append(delta)

        # Reverse deltas to match layer order
        deltas.reverse()

        # Update weights and biases
        for i in range(len(self.weights)):
            layer_input = X if i == 0 else self.activations[i - 1]
            self.weights[i] -= learning_rate * np.dot(layer_input.T, deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0)

    def train(self, X_train, y_train, batch_size, learning_rate, epochs):
        for epoch in range(epochs):
            self.save_to_file("nn_weights_biases.txt")
            # Shuffle the dataset
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            # Iterate over mini-batches
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Update the network weights and biases using the mini-batch
                self.backpropagate(X_batch, y_batch, learning_rate)


# Training parameters
learning_rate = 0.01
epochs = 10000

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape the input data
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Convert the labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Create and train a neural network
layer_sizes = [784, 128, 100, 10]
nn = CustomNeuralNetwork(layer_sizes)
# if neural net loading doesn't work, remove this
try:
    nn.load_from_file("nn_weights_biases.txt")
    nn.forward(x_test)
except:
    print("network could not be read from file")
# -+-nn.train(x_train, y_train, batch_size=10, learning_rate=0.05, epochs=25)


# Test the trained network
y_pred = nn.forward(x_test)
y_true = np.argmax(y_test, axis=1)
accuracy = np.mean(y_true == y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Function to display the images and their labels
def display_images(images, true_labels, pred_labels, cmap=plt.cm.binary):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=cmap)
        plt.xlabel(f"True: {true_labels[i]}, Pred: {find_highest(pred_labels[i])}")
    plt.show()


# Reshape the test images back to their original format
x_test_reshaped = x_test.reshape(x_test.shape[0], 28, 28)

# Display the images along with their true and predicted labels
display_images(x_test_reshaped, y_true, y_pred)

