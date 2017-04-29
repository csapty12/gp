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


# def split_data(data, labels, train_s =0.8, test_s = 0.2):
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

def cross_validation(K, n_val, activation_f, solver_f, data_CBD, class_labels_CBD):
    n_range_CBD = range(1,51)
    n_scores_CBD = list()
    for n in n_range_CBD:
        if n_val == 1:
            mlp_CBD = MLPClassifier(activation = activation_f, solver=solver_f, alpha=1e-5,hidden_layer_sizes=(n))
        elif n_val == 2:
            mlp_CBD = MLPClassifier(activation = activation_f, solver=solver_f, alpha=1e-5,hidden_layer_sizes=(n,n))
        elif n_val == 3:
            mlp_CBD = MLPClassifier(activation = activation_f, solver=solver_f, alpha=1e-5,hidden_layer_sizes=(n,n,n))
        scores_CBD = cross_val_score(mlp_CBD,data_CBD,class_labels_CBD, cv=K, scoring = "accuracy")

        n_scores_CBD.append(scores_CBD.mean())

    print(n_scores_CBD)
    print("mean: ")
    print(sum(n_scores_CBD)/len(n_scores_CBD))
    return n_scores_CBD



def find_opt_nodes(n_scores):
    max_num = max(n_scores)
    for i in range(len(n_scores)):
        if n_scores[i] == max_num:
            print("number nodes: ", i+1)
            print("accuracy: ", max_num*100, "%")
            print("accuracy: ", round(max_num*100,2),"%")
            return i+1
        

if __name__ == "__main__":
	x = read_data('./dataset2.txt')

	data = x[0]
	labels = x[1]

	test1 = cross_validation(10, 1, 'logistic', 'lbfgs', data, labels)
	test2 = cross_validation(10, 2, 'logistic', 'lbfgs', data, labels)
	test3 = cross_validation(10, 3, 'logistic', 'lbfgs', data, labels)
	n_CBD1 = find_opt_nodes(test1)
	n_CBD2 = find_opt_nodes(test2)
	n_CBD3 = find_opt_nodes(test3)

	print(n_CBD1)
	print(n_CBD2)
	print(n_CBD3)

	# plt.figure()
	# # figsize(8,8)
	# plt.plot(range(1,51),test1,"b", label ="Cross Validation Accuracy 1")
	# plt.show()



	plt.figure()
	plt.boxplot([test1, test2, test3])
	plt.ylabel("average predictive accuracy")
	plt.xlabel("n number of hidden layers")
	plt.show()



