# SimpleNeuralNet
A simple neural network model with backpropagation. There is a file for recognizing handwritten digits using the MNIST dataset, as well as a basic file that does not use tensorflow and is more customizable for any task.

# Use
Feel free to try it out yourself! The code for the handwritten digits was made in pycharm, so it has not been tested with other editors, but the code should work just fine so long as you have the proper modules installed. The basic net is much more customizable and can be used without the need for many of the other modules, just numpy and json. I've been trying to work on a version that doesn't use any outside modules, but it would be a lot slower and the code is much more complicated to write.

To train the network, simply use the train function with your own datasets as well as some parameters such as learning rate, batch size, and epochs.

Important note: The basic net comes with a save/load system by default that will try to read a text file for the weights and biases. If you do not have a text file set up, simply make one in the project location and name it properly, or just remove the part of the code that attempts to read the file.

<img src="https://user-images.githubusercontent.com/107130695/235031900-0c53f4f7-45a3-461d-a416-313c5df138b7.png" height="256px" width="256px">
Recognizing handwritten digits (It really struggles with that one 5)

# Contribution
You can share any code improvements or models by contributing! If you want to add your own trained model, you can make a folder with the weights and biases and/or python code. I'm not sure what will become of this project, but contribution is always welcome :)
