from functions import *
import numpy as np
import tensorflow as tf
import os

class Neural_Network:
    def sigmoid(self, weighted_sum: float):
        # Sigmoid function: S(w) = 1/(1 + e⁻ʷ)
        return 1/(1 + np.exp(-weighted_sum))

    def squared_loss(self, target, prediction):
        return np.subtract(target, prediction) ** 2

    def weighted_sum(self, matrix1, matrix2, bias: int):
        # Here we use the matrix's dot product ∑aᵢ*bᵢ, then add the bias of each neuron to the result
        return np.dot(matrix1, matrix2) + bias

    def derrivative_error(self, input, target):
        # Calculate the error for every neuron
        self.delta = []
        delta_layer = []
        
        # Error output neuron
        for i in range(self.number_output):
            delta_i = self.result_value[i] * (1 - self.result_value[i]) * (self.result_value[i] - target[i])
            delta_layer.append(delta_i)
        self.delta.insert(0, delta_layer)
        
        # Error n-hidden layer
        delta_layer = []
        weight_layer = self.get_weight(self.hidden_layer) 
        for i in range(self.neuron_per_layer):
            delta_i = self.neuron_value[-1][i] * (1 - self.neuron_value[-1][i])
            summation = 0
            for connection in range(self.number_output):
                summation += weight_layer[connection][i] * self.delta[0][connection] # Since weights len is always number of hidden layer + 1
            delta_i *= summation
            delta_layer.append(delta_i)
        self.delta.insert(0, delta_layer)

        # Error n-1-hidden layer to 1-hidden layer
        for layer in reversed(range(self.hidden_layer - 1)):
            delta_layer = []
            weight_layer = self.get_weight(layer + 1)   # Because there are hidden layer + 1 element in array, and here layer will always be 2 less than the index
            for i in range(self.neuron_per_layer):
                delta_i = self.neuron_value[layer][i] * (1 - self.neuron_value[layer][i])
                summation = 0
                for connection in range(self.neuron_per_layer):
                    summation += weight_layer[connection][i] * self.delta[0][connection]
                delta_i *= summation
                delta_layer.append(delta_i)
            self.delta.insert(0, delta_layer)

    def forward_propagation(self, input_data):
        input = input_data

        # Get weight for first layer
        weight_layer = self.get_weight(0)
        bias_layer = self.get_bias(0)

        # Calculate value to first hidden layer from input using the weighted sum
        # Pass the calculated value to activation function (sigmoid function)
        # Add the value of activation function to an array
        activation_value = []
        for neuron in range(self.neuron_per_layer):
            weighted_sum =  self.weighted_sum(input, weight_layer[neuron], bias_layer[neuron])
            activation = self.sigmoid(weighted_sum)
            activation_value.append(activation)

        # Add the activation value array to neuron_value array
        self.neuron_value[0] = activation_value

        # Calculate the value of neuron for the next hidden layer till the last hidden layer
        for layer in range(1, self.hidden_layer):
            input = activation_value
            activation_value = []
            weight_layer = self.get_weight(layer)
            bias_layer = self.get_bias(layer)
            for neuron in range(self.neuron_per_layer):
                weighted_sum =  self.weighted_sum(input, weight_layer[neuron], bias_layer[neuron])
                activation = self.sigmoid(weighted_sum)
                activation_value.append(activation)
            self.neuron_value[layer] = activation_value

        # Calculate the value of neuron for the output layer
        input = activation_value
        activation_value = []
        weight_layer = self.get_weight(self.hidden_layer)   # Weight on the last index which is the amount of hidden layer
        bias_layer = self.get_bias(self.hidden_layer)
        for neuron in range(self.number_output):
            weighted_sum = self.weighted_sum(input, weight_layer[neuron], bias_layer[neuron])
            activation = self.sigmoid(weighted_sum)
            activation_value.append(activation)
        self.result_value = activation_value
        
    def backward_propagation(self, input_data, target_data):
        input = input_data
        target = target_data

        # Calculate every neuron's error
        self.derrivative_error(input, target)

        # Calculate new weight & new bias
        for layer in reversed(range(len(self.weights))):
            for i in range(len(self.weights[layer][0])):
                for j in range(len(self.weights[layer])):
                    self.weights[layer][j][i] -= (self.learn_rate * self.delta[layer][j] * (input[i] if layer == 0 else self.neuron_value[layer - 1][i]))
                    self.bias[layer][j] -= (self.learn_rate * self.delta[layer][j] * 1)

    def train_network(self, epoch: int):       
        # To get the statistic of the data 
        self.train_losses = []
        self.train_accuracy = []

        # Divide the train data into 20 batch, each contains 3000 data to train
        for i in range(20):
            index = i * 3000
            printProgressBar(0, 3000, prefix = f'Batch {i + 1}/20:', suffix = 'Complete', length = 50, is_test=False)
            batch_losses = []
            batch_accuracy = []
            for j in range(3000):
                # Get the current input data
                input = self.train_data_input[index + j]
                input = self.resize_input(input)
                input.flatten()

                # Get the current target data
                target = to_list(self.train_data_target[index + j])

                # Pass the input data to forward propagation algorithm
                self.forward_propagation(input)
                
                # Calculate the losses for each data, using the function squared loss
                prediction = np.array(self.result_value)
                batch_losses.append(self.squared_loss(target, prediction))

                # To check if we got correct prediction during the training process
                batch_accuracy.append(1 if self.check_prediction(prediction, target) else 0)

                # Pass the input and target data to backward propagation algorithm
                self.backward_propagation(input,target)
                printProgressBar(j + 1, 3000, prefix = f'Batch {i + 1}/20:', suffix = 'Complete', length = 50, is_test=False)
            self.train_losses.append(np.mean(np.array(batch_losses)) * 100)
            self.train_accuracy.append(np.mean(np.array(batch_accuracy)) * 100)
            # print(f"Training batch {i + 1}: Accuracy: {(np.mean(np.array(batch_accuracy)) * 100):.3f} % , Loss: {(np.mean(np.array(batch_losses)) * 100):.3f} %")
        visualize_growth(accuracy=self.train_accuracy, loss=self.train_losses,x_param=[(batch+1) for batch in range(20)], file_name=f"Batch_Growth_E{epoch + 1}.png", isShowing=False)
        print()
        # Final message    
        print(f"""
            Succesfully trained the network!
            Average accuracy: {(np.mean(np.array(self.train_accuracy))):.3f} %
            Average losses: {(np.mean(np.array(self.train_losses))):.3f} %
        """)
    
    def save_training_params(self):
        cur_dir = os.path.abspath(os.getcwd())
        
        # Save the number of weights layer
        file_path = os.path.join(cur_dir, "parameter", 'number_w_layer.txt')
        f = open(file_path, 'w')
        f.write(str(self.get_w_b_layer(True)))
        f.close()

        # Save each layer of weights (since numpy have method to save numpy.array to txt file and load it instantly)
        for i in range(self.get_w_b_layer(is_weight = True)):
            file_title = f"weight_layer{i}.txt"
            text_path = os.path.join(cur_dir, "parameter", file_title)
            np.savetxt(text_path, self.get_weight(i), delimiter=',')
        
        # Save the number of bias layer
        file_path = os.path.join(cur_dir, "parameter", 'number_b_layer.txt')
        f = open(file_path, 'w')
        f.write(str(self.get_w_b_layer(is_weight = False)))
        f.close()

        # Save each layer of bias (since numpy have method to save numpy.array to txt file and load it instantly)
        for i in range(self.get_w_b_layer(False)):
            file_title = f"bias_layer{i}.txt"
            text_path = os.path.join(cur_dir, "parameter", file_title)
            np.savetxt(text_path, self.get_bias(i), delimiter=',')

    def check_prediction(self, prediction, target):
        prediction_res = np.argmax(prediction)
        target_res = np.argmax(target)
        return prediction_res == target_res

    def set_test_weights_bias(self):
        cur_dir = os.path.abspath(os.getcwd())
        file_title = "number_w_layer.txt"
        text_path = os.path.join(cur_dir, "parameter", file_title)
        f = open(text_path, "r")
        number_weight = int(f.readline())
        f.close()

        for i in range(number_weight):
            file_title = f"weight_layer{i}.txt"
            text_path = os.path.join(cur_dir, "parameter", file_title)
            self.weights[i] = np.loadtxt(text_path, delimiter=',')

        cur_dir = os.path.abspath(os.getcwd())
        file_title = "number_b_layer.txt"
        text_path = os.path.join(cur_dir, "parameter", file_title)
        f = open(text_path, "r")
        number_bias = int(f.readline())
        f.close()

        for i in range(number_bias):
            file_title = f"bias_layer{i}.txt"
            text_path = os.path.join(cur_dir, "parameter", file_title)
            self.bias[i] = np.loadtxt(text_path, delimiter=',')

    def test_network(self):
        self.set_test_weights_bias()

        self.test_losses = []
        self.test_accuracy = []

        for batch in range(10):
            batch_losses = []
            batch_accuracy = []
            index = 1000 * batch
            printProgressBar(0, 1000, prefix = f'Batch {batch + 1}/20:', suffix = 'Complete', length = 50, is_test = True)
            for j in range(1000):
                # Get the current input data
                input = self.test_data_input[index + j]
                input = self.resize_input(input)
                input.flatten()
                
                # Get the current target data
                target = to_list(self.test_data_target[index + j])

                # Pass the input data to forward propagation algorithm
                self.forward_propagation(input)

                # Calculate the losses for each data, using the function squared loss
                prediction = np.array(self.result_value)
                batch_losses.append(self.squared_loss(target, prediction))

                # To check if we got correct prediction during the training process
                batch_accuracy.append(1 if self.check_prediction(prediction, target) else 0)

                printProgressBar(j + 1, 1000, prefix = f'Batch {batch + 1}/20:', suffix = 'Complete', length = 50, is_test = True)
            self.test_losses.append(np.mean(np.array(batch_losses))* 100)
            self.test_accuracy.append(np.mean(np.array(batch_accuracy)) * 100)
            print(f"Test batch {batch + 1}: Accuracy: {(np.mean(np.array(batch_accuracy)) * 100):.3f} % , Loss: {(np.mean(np.array(batch_losses)) * 100):.3f} %")
        # Final message    
        print(f"""
            Succesfully tested the network!
            Average accuracy: {(np.mean(np.array(self.test_accuracy))):.3f} %
            Average losses: {(np.mean(np.array(self.test_losses))):.3f} %
        """)
    
    def predict(self, input_data):
        # input = self.resize_input(input_data)           # Resize the input data to 10 x 10 matrix
        input = input_data.flatten()                                  # Flatten the array to 100 sized array

        self.forward_propagation(input)

        prediction = np.array(self.result_value)

        return np.argmax(prediction), round(np.amax(prediction) * 100, 3)

    def get_weight(self, layer: int):
        # To get the array of weights layer
        return self.weights[layer]

    def get_bias(self, layer: int):
        # To get the array of bias layer
        return self.bias[layer]

    def get_w_b_layer(self, is_weight: bool):
        # Return the length of weights(bias)/ the number layer of weights(bias)
        return len(self.weights) if is_weight else len(self.bias)

    def get_average_train_statistics(self):
        # Return the train accuracy and losses for each batch
        return np.mean(np.array(self.train_accuracy)), np.mean(np.array(self.train_losses))

    def resize_input(self, original_pixel):
        # To resize the input data array to the desired size or to resize the pixel from input data to the desired size
        image = tf.constant(original_pixel)
        image = image[tf.newaxis, ..., tf.newaxis]

        image.shape.as_list()

        return tf.image.resize(image, self.pixel_size).numpy().flatten()

    def random_weight_bias(self, row: int, column: int, epsilon: float):
        # To initialize weight values between (-epsilon, epsilon)
        return np.random.rand(row, column) * (2 * epsilon) - epsilon

    def visualize_growth(self):
        plt.figure().set_size_inches(13, 8)
        plt.plot()
        print(self.test_accuracy)
        print(self.test_losses)
    
    def __init__(self, pixel_size: tuple[int, int], learn_rate: float, hidden_layer: int, neuron_per_layer: int) -> None:
        self.pixel_size = pixel_size
        self.number_input = self.pixel_size[0] * self.pixel_size[1]
        self.number_output = 10 # Because there are 10 digits 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

        self.hidden_layer = hidden_layer
        self.neuron_per_layer = neuron_per_layer

        self.weights = []
        # Add weights for every connection between input neuron and first hidden layer
        # Range, R(-√6/√(fan-in+fan-out, √6/√(fan-in+fan-out) fan-in: Number of input to neuron, fan-out: Number of output from neuron
        eps = 2 # np.sqrt(6)/np.sqrt(self.number_input + self.neuron_per_layer)
        self.weights.append(self.random_weight_bias(self.neuron_per_layer, self.number_input, eps))
        # Add weights for every connection between i-hidden layer and i+1-hidden layer, for layer 1 till n-1
        for _ in range(self.hidden_layer - 1):
            eps = 2 # np.sqrt(6)/np.sqrt(self.neuron_per_layer + self.neuron_per_layer)
            self.weights.append(self.random_weight_bias(self.neuron_per_layer, self.neuron_per_layer, eps))
        # Add weights for every connection between n-hidden layer and output neuron
        eps = 2 # np.sqrt(6)/np.sqrt(self.neuron_per_layer + self.number_output)
        self.weights.append(self.random_weight_bias(self.number_output,self.neuron_per_layer, eps))

        self.bias = []
        # Add bias for every neuron in every hidden layer
        # For bias we set the initial bias are 0
        for _ in range(self.hidden_layer):
            eps = 5 # np.sqrt(6)/np.sqrt(self.neuron_per_layer + self.neuron_per_layer)
            self.bias.append(self.random_weight_bias(1, self.neuron_per_layer, eps).flatten())
        # Add bias for every neuron in the output layer
        eps = 5 # np.sqrt(6)/np.sqrt(self.neuron_per_layer + self.number_output)
        self.bias.append(self.random_weight_bias(1,self.number_output, eps).flatten())

        self.learn_rate = learn_rate

        self.train_data_input, self.train_data_target = get_data("train")
        self.test_data_input, self.test_data_target = get_data("test")
        
        # To save the value of result and neuron value for each data in train and test data, for the forward and backward propagation
        self.neuron_value = [[] for _ in range(self.hidden_layer)]
        self.result_value = []