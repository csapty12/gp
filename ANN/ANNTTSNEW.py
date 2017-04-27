import numpy as np
from numpy.random import shuffle
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import timeit
import matplotlib.pyplot as plt
import sys


import wget

def read_data(file_name):
	# load data from file
	CBD=np.loadtxt(file_name)
	CBD = CBD # to keep data consistent 
	shuffle(CBD)

	class_labels_CBD =CBD[:,-1]
	class_labels_CBD= [ int(x) for x in class_labels_CBD ]
	class_labels_CBD = np.asarray(class_labels_CBD)

	data_CBD=CBD[:,0:-1]

	return data_CBD, class_labels_CBD

def split_data(data,labels, train_s, test_s):
	X_train, X_test, y_train, y_test = train_test_split(data,labels, train_size=train_s,test_size=test_s)
	return X_train, y_train,X_test, y_test


def run_classifier(x_train, y_train, struct, x_test,  activate='logistic', iter_limit=1000):
	mlp = MLPClassifier(hidden_layer_sizes=struct, activation=activate, solver='lbfgs', max_iter=iter_limit)
	mlp.fit(x_train,y_train)
	predictions = mlp.predict(x_test)

	return predictions

def predictive_accuracy(predictions, y_test):
	trufa = y_test==predictions
	accuracy = round((sum(trufa)/len(trufa))*100)
	# print("Accuracy: ", accuracy ,"%")
	return accuracy

def run(training_size, testing_size):
	
	read_d = read_data('banking.txt')
	data = read_d[0]
	labels = read_d[1]

	accuracies = list()
	times = list()
	for i in range(1000):
		start_time = timeit.default_timer()
		x = split_data(data,labels, training_size, testing_size)
		x_train = x[0]
		y_train = x[1]
		x_test = x[2]
		y_test = x[3]


		train_model = run_classifier(x_train,y_train, (5,5), x_test)
		# print(train_model)
		elapse = timeit.default_timer() - start_time
		times.append(elapse)
		accuracy = predictive_accuracy(train_model, y_test)
		# print("predictive accuracy of model: ", accuracy)
		accuracies.append(accuracy)

		if i % 100==0:
			# print("iteration: ", i)
			sys.stdout.write("iteration: "+str(i)+"\n" )

	# print(accuracies)
	# print("time taken")
	# print(times)

	# plt.boxplot(accuracies)
	
	# plt.figure()
	# plt.plot(list(range(len(times))), times, label = "Time taken to learn")
	# plt.legend(loc = "best")
	# plt.show()
	# plt.plot(list(range(len(accuracies))), accuracies, "b", label= "predictive accuracies over 2000 iterations")
	# plt.legend(loc= "best")

	save_file = open("./out.txt", 'a')
	for i in accuracies:
	    save_file.write(str(i))
	    save_file.write(", ")

	save_file.write("\n")
	save_file.close()
	sys.stdout.write("\n")
	







if __name__=="__main__":
	run(0.9,0.1)




