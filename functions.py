from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def to_list(target: int):
    return np.array([1 if target == i else 0 for i in range(10)])


def get_data(type: str):
    # Retrieve the data
    data = mnist.load_data() # ((input_train, output_train), (input_test, output_test))

    # Train data is on the first element of tuple
    X_train = data[0][0]/255
    y_train = data[0][1]
    
    # Test data is on the second element of tuple
    X_test = data[1][0]/255
    y_test = data[1][1]

    if type == "train":
        # Return the training data
        return X_train, y_train
    elif type == "test":
        return X_test, y_test

def visualize_growth(accuracy, loss, x_param, file_name, isShowing: bool = False):
    plt.figure()
    plt.plot(x_param, accuracy, color='#820401', marker='o', label='Average accuracy')
    plt.plot(x_param, loss, color='#29066B', marker='o', label='Average loss')
    plt.xticks(x_param)
    plt.yticks([(index + 1) * 5 for index in range(20)])
    plt.legend(loc='lower center')
    plt.title('Artificial Neural Network')
    plt.grid()
    cur_dir = os.path.abspath(os.getcwd())
    if isShowing:
        file_path = os.path.join(cur_dir, "graph", "epochs", file_name)
    else:
        file_path = os.path.join(cur_dir, "graph", "batches", file_name)
    plt.savefig(file_path)
    if isShowing:
        plt.show()

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r", is_test = False):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total and is_test:
        print()