import module as nn
import numpy as np
import random

with open("dataset.txt", "r", encoding="utf-8") as f:
    dataset = f.read()

chars = sorted(list(set(dataset)))
charlen = len(chars)
character_dict = {character: index for index, character in enumerate(chars)}  # dict of possible chars
reverse_dict = {index: character for index, character in enumerate(chars)}


def encode_text(text):
    return [character_dict[i] for i in text]


def decode_text(values):
    return ''.join([reverse_dict[i] for i in values])


train_threshold = int(len(dataset) * 0.9)
train_data = dataset[:train_threshold]
test_data = dataset[train_threshold:]
print(character_dict, reverse_dict)
block_size = 8
layer_sizes = [block_size, 728, 512, 512, charlen]
network = nn.CustomNeuralNetwork(layer_sizes, activation_function='tanh')


def one_hot_encode(n, length):
    one_hot = np.zeros(length)
    one_hot[n] = 1
    return one_hot


def get_batch():
    random_index = random.randint(1, len(train_data) - (2 * block_size + 1))
    return encode_text(train_data[random_index:random_index + (2 * block_size) + 2])


def data_from_batch(batch):
    xb = [batch[block_size:block_size * 2]]
    yb = [one_hot_encode(batch[block_size * 2], charlen)]
    for x in range(block_size):
        block_arr = batch[x + 1:block_size + x + 1]
        """for y in range(block_size - (x + 1)):
            block_arr.append(0)"""
        xb.append(np.asarray(block_arr))
        yb.append(one_hot_encode(batch[x], charlen))
    xb = [np.asarray(x).reshape(-1, 1) for x in xb]
    return xb, yb


def train(learning_rate, epochs):
    for epoch in range(epochs):
        input_batch, output_batch = data_from_batch(get_batch())
        for b in range(1, 10):
            for x, y in zip(input_batch, output_batch):
                x = np.asarray(x).T
                network.backpropagate(x, y, learning_rate)


def find_highest(arr):
    highest = 0
    index = 0
    for i, v in enumerate(arr):
        if v > highest:
            highest = v
            index = i
    return int(index)


def pad_input(input, size):
    return np.pad(np.asarray(input), (0, size - len(input)), 'constant')


def prompt(text, length=100):
    inputtxt = text
    outputtxt = inputtxt
    for x in range(length):
        encoded = encode_text(inputtxt)
        padded = pad_input(encoded, block_size)
        prediction = network.forward(padded)
        highest = find_highest(prediction)
        decoded = decode_text([highest])
        outputtxt += decoded
        inputtxt += decoded
        if len(inputtxt) >= block_size:
            inputtxt = inputtxt[1:]
    return outputtxt

"""try:
    network.load_from_file("GPTweights_biases.txt")  # rename to whatever your text file is
except:
    print("network could not be read from file")"""

train(0.003, 50)
network.save_to_file("GPTweights_biases.txt")
print(prompt("Hello"))
