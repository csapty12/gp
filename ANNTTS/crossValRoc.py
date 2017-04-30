import numpy as np
from numpy.random import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

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

def K_fold_optimal_model2(data, labels, splits, optimal_dim):
	cv = StratifiedKFold(n_splits=splits)
	classifier = MLPClassifier(hidden_layer_sizes =optimal_dim, activation="logistic", solver ='lbfgs')
	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)

	colors = cycle(['cyan', 'indigo', 'seagreen'])
	lw = 2
	i = 0
	for (train, test), color in zip(cv.split(data, labels), colors):
	    probas_ = classifier.fit(data[train], labels[train]).predict_proba(data[test])
	    # Compute ROC curve and area the curve
	    fpr, tpr, thresholds = roc_curve(labels[test], probas_[:, 1])
	    mean_tpr += interp(mean_fpr, fpr, tpr)
	    mean_tpr[0] = 0.0
	    roc_auc = auc(fpr, tpr)

	mean_tpr /= cv.get_n_splits(data, labels)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	return mean_fpr, mean_tpr, mean_auc

def draw_mean_optimal_ROC(means):
	m1 = means[0]
	m2 = means[1]
	m3 = means[2]


	plt.plot(m1[0], m1[1], color='g', linestyle='-',
	        label='(5 49 1) Opt. Mean (AUC = %0.2f)' % m1[2], lw=2)
	plt.plot(m2[0], m2[1], color='b', linestyle='--',
	        label='(5 36 36 1) Opt. Mean (AUC = %0.2f)' % m2[2], lw=2)
	plt.plot(m3[0], m3[1], color='orange', linestyle='-',
	        label='(5 37 37 37 1) Opt. Mean ROC (AUC = %0.2f)' % m3[2], lw=2)
	plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
		        label='Random Luck')
	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC rurve to represent Optimal Nodes against True Positive Rate')
	plt.legend(loc="lower right")
	plt.show()
	


if __name__=="__main__":
	x =read_data('./dataset2.txt')
	data = x[0]
	labels = x[1]
	# optimal nodes taken from 
	opt_func1 = K_fold_optimal_model2(data,labels,10,(49))
	print(opt_func1)
	opt_func2 = K_fold_optimal_model2(data,labels,10,(36,36))
	opt_func3 = K_fold_optimal_model2(data,labels,10,(37,37,37))


	mean1 = (opt_func1[0],opt_func1[1],opt_func1[2],'g')
	mean2 = (opt_func2[0],opt_func2[1],opt_func2[2],'b')
	mean3 = (opt_func3[0],opt_func3[1],opt_func3[2],'r')

	l = [mean1, mean2, mean3]
	draw_mean_optimal_ROC(l)



	