import numpy as np
from numpy.random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import timeit
import sys
import configparser
import os.path
import pandas as pd
from ggplot import *
import matplotlib.pyplot as plt


def make_config():
    """
    If a configuration file does not exist, then create a default one with default parameters.
    :return: the configuration file
    """
    config = configparser.ConfigParser()
    config['DEFAULT'] = {
        'test_size': '0.2',
        'train_size': '0.2',
        'structure': '5 5',
        'activation': 'logistic',
        'learning_method': 'lbfgs',
        'max_iterations': '1000'
    }
    with open('config.ini', 'w') as configfile:
        config.write(configfile)


def read_config_file(file):
    """
    function to read the configuration file which takes the parameters and is used to train and test the ANN.
    :param file: the configuration file
    :return: the configuration parameters.
    """
    config = configparser.ConfigParser()
    config.read(file)
    params = list()
    # get the key values for each parameter
    train_size = config['DEFAULT']['train_size']
    test_size = config['DEFAULT']['test_size']
    ANN_structure = config['DEFAULT']['structure']
    learning_method = config['DEFAULT']['learning_method']
    activation = config['DEFAULT']["activation"]
    max_iter = config['DEFAULT']["max_iterations"]
    params.append(train_size)
    params.append(test_size)
    params.append(ANN_structure)
    params.append(activation)
    params.append(learning_method)
    params.append(max_iter)
    return params


def read_data(file_name):
    """
    Function to read in the data file used for classification
    :param file_name: the file to be read in for the ANN model
    :return: the shuffled data set and its associated classification labels.
    """
    # load data from file
    CBD = np.loadtxt(file_name)
    CBD = CBD  # to keep data consistent
    shuffle(CBD)

    class_labels_CBD = CBD[:, -1]
    class_labels_CBD = [int(x) for x in class_labels_CBD]
    class_labels_CBD = np.asarray(class_labels_CBD)

    data_CBD = CBD[:, 0:-1]
    return data_CBD, class_labels_CBD


def split_data(data, labels, train_s, test_s):
    """
    Function to randomly split the data into a training and testing dataset. X_train and y_train represent the
    data used to train the model. X_test and y_test represent the data used to test the predictive accuracy of the
    model. Here, X represents each row of data, and y represents the classification label associated with a row of data.
    :param data: the data set without the classification labels
    :param labels:  the classification label array
    :param train_s: the percentage of data used to train the model
    :param test_s: the percentage of data used to test the model
    :return:
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=train_s, test_size=test_s)
    return X_train, y_train, X_test, y_test


def run_classifier(x_train, y_train, struct, x_test, activate='logistic', solve='lbfgs', iter_limit=1000):
    """
    Function to train and test the ANN classifier model once using the training and testing data.
    :param x_train: the training data
    :param y_train: the training data classification labels
    :param struct: the structure of the ANN, given by the config.txt file
    :param x_test: the data to be used to test the predictive accuracy of the model
    :param activate: the type of activation function used to transfer data from one node to the next
    :param solve: The type of back propagation algorithm used.
    :param iter_limit: the maximum iteration limit given to the model given by the user
    :return: the predictions after training and testing the model
    """
    mlp = MLPClassifier(hidden_layer_sizes=struct, activation=activate, solver=solve, max_iter=iter_limit)
    mlp.fit(x_train, y_train)
    predictions = mlp.predict(x_test)

    return predictions


def predictive_accuracy(predictions, y_test):
    """
    Function to return the predictive accuracy of the model after testing on the hold out sample data.
    :param predictions: the predictions from testing
    :param y_test: the actual testing classification labels
    :return: the percentage accuracy of the ANN model
    """
    trufa = y_test == predictions
    # print(confusion_matrix(y_test,predictions))

    accuracy = round((sum(trufa) / len(trufa)) * 100)
    return accuracy


def run(data_set, config_file, show_graph=False, boxplt=False, roc=False):
    """
    Function to run the ANN model to train and test the model.
    :param data_set: the data set to be used to train and test the model
    :param config_file: the configuration file to determine the ANN structure
    :param run_n_times: the number of times to run the model
    :return: the accuracy of the trained model on the testing data set in a out.txt file.
    """

    # get the structure of the ANN from the configuration file
    train_size = float(config_file[0])
    test_size = float(config_file[1])
    hidden_layers = tuple(map(int, config_file[2].split(' ')))
    activation = config_file[3]
    max_iterations = int(config_file[5])

    accuracies = list()
    times = list()
    for i in range(max_iterations):
        # reshuffle the datasets each time
        read_d = read_data(data_set)
        data = read_d[0]
        labels = read_d[1]
        start_time = timeit.default_timer()
        x = split_data(data, labels, train_size, test_size)
        x_train = x[0]
        y_train = x[1]
        x_test = x[2]
        y_test = x[3]
        train_model = run_classifier(x_train, y_train, hidden_layers, x_test, activate=activation)
        predsTESTING = train_model[1]
        # print(predsTESTING)
        elapse = timeit.default_timer() - start_time
        times.append(elapse)
        accuracy = predictive_accuracy(train_model, y_test)
        accuracies.append(accuracy)
        if i % 100 == 0:
            sys.stdout.write("iteration: " + str(i) + "\n")

    if show_graph == True and roc == True:
        False_Positive_Rate, True_Positive_Rate, _ = metrics.roc_curve(y_test, predsTESTING)
        df = pd.DataFrame(dict(fpr=False_Positive_Rate, tpr=True_Positive_Rate))
        auc = metrics.auc(False_Positive_Rate, True_Positive_Rate)
        g = ggplot(df, aes(x='False_Positive_Rate', y='True_Positive_Rate')) + geom_line() + geom_abline(
            linetype='dashed') + geom_area(alpha=0.2) + \
            ggtitle("ROC Curve w/ AUC=%s" % str(auc))
        g.show()

    if show_graph == True and boxplt == True:
        plt.boxplot(accuracies)
        plt.figure()
        plt.plot(list(range(len(times))), times, label="Time taken to learn")
        plt.legend(loc="best")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Time taken (s)")
        plt.show()
        plt.plot(list(range(len(accuracies))), accuracies, "b", label="predictive accuracies over 2000 iterations")
        plt.legend(loc="best")
        plt.show()

    save_file = open("./out.txt", 'a')
    for i in accuracies:
        save_file.write(str(i))
        save_file.write(", ")

    save_file.write("\n")
    save_file.close()
    sys.stdout.write("\n")


if __name__ == "__main__":
    file_path = './config.ini'
    if os.path.exists(file_path):
        configuration_file = read_config_file(file_path)
    else:
        print("No config file detected, creating default config.ini file")
        make_config()
        configuration_file = read_config_file(file_path)
    run('./dataset2.txt', configuration_file, show_graph=False)
