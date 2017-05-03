import numpy as np
from numpy.random import shuffle
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


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


def MLP_cross_val_classifier(data_set, K, n_val, activation_f='logistic', solver_f='lbfgs'):
    data_s = read_data(data_set)
    data = data_s[0]
    class_labels = data_s[1]
    n_range = range(1, 51)
    n_scores = list()
    for n in n_range_CBD:
        if n_val == 1:
            mlp = MLPClassifier(activation=activation_f, solver=solver_f, alpha=1e-5, hidden_layer_sizes=(n))
        elif n_val == 2:
            mlp = MLPClassifier(activation=activation_f, solver=solver_f, alpha=1e-5, hidden_layer_sizes=(n, n))
        elif n_val == 3:
            mlp = MLPClassifier(activation=activation_f, solver=solver_f, alpha=1e-5, hidden_layer_sizes=(n, n, n))
        scores = cross_val_score(mlp, data, class_labels, cv=K, scoring="accuracy")
        n_scores.append(scores.mean())

    print(n_scores)
    print("mean: ")
    print(sum(n_scores) / len(n_scores))
    return n_scores


def find_opt_nodes(n_scores):
    max_num = max(n_scores)
    for i in range(len(n_scores)):
        if n_scores[i] == max_num:
            print("number nodes: ", i + 1)
            print("accuracy: ", max_num * 100, "%")
            print("accuracy: ", round(max_num * 100, 2), "%")
            return i + 1


test1 = cross_validation(10, 1, 'logistic', 'lbfgs')
test2 = cross_validation(10, 2, 'logistic', 'lbfgs')
test3 = cross_validation(10, 3, 'logistic', 'lbfgs')
n_CBD1 = find_opt_nodes(test1)
n_CBD2 = find_opt_nodes(test2)
n_CBD3 = find_opt_nodes(test3)


def run(data_set):
    cross_val_mean = MLP_cross_val_classifier(data_set, 2, 1)
    False_Positive_Rate, True_Positive_Rate, _ = metrics.roc_curve(y_test, predsTESTING)


import matplotlib.pyplot as plt

figure()
plt.plot(range(1, 51), test1, "b", label="Cross Validation Accuracy 1")
plt.xlabel("number of possible nodes in the network")
plt.ylabel("cross val accuracy")
legend(loc="best")

figure()
plt.plot(range(1, 51), test2, "g", label="Cross Validation Accuracy 2")
plt.xlabel("number of possible nodes in the network")
plt.ylabel("cross val accuracy")
legend(loc="best")

figure()
plt.plot(range(1, 51), test3, "r", label="Cross Validation Accuracy 3")
plt.xlabel("number of possible nodes in the network")
plt.ylabel("cross val accuracy")
legend(loc="best")

if __name__ == "__main__":
    run('./dataset2.txt')
