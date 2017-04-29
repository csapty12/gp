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

def MLP_cross_val_classifier(data_set,K, n_val, activation_f='logistic', solver_f='lbfgs'):
	data_s = read_data(data_set)
	data_CBD = data_s[0]
	class_labels_CBD= data_s[1]
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

def run(data_set):
	cross_val_mean = MLP_cross_val_classifier(data_set,2,1)
	False_Positive_Rate, True_Positive_Rate, _ = metrics.roc_curve(y_test, predsTESTING)


if __name__ =="__main__":
	run('./dataset2.txt')