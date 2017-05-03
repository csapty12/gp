#==========================================CONFIG FILE======================================================
#How to use:
#   Please enter the configuration details of how you would like to construct the artificial neural network
#   the structure is as follows:

#   :param structure: for every hidden layer, insert value for number of nodes per hidden layer, seperated by a space
#   :param learning_method: Learning Methods include -> choice {lbfgs, sgd, adam}
#   :param test_size: percentage of dataset to test the network (value between 0 and 1)
#   :param train_size: percentage of dataset to train the network (value between 0 and 1)
#   -> please ensure that values do not exceed 1 or smaller than 0
#   -> for more relastic predicitons, use a training dataset percentage of size x and testing dataset
#   size 1-x
#   :param activation: activation function to be used -> choice {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
#   :param max_itearions: maximum number of iterations to be used before learning MUST stop.

Example:


[DEFAULT]
structure = 5 5
learning_method = lbfgs
test_size = 0.2
train_size = 0.8
activation = logistic
max_iterations = 1000

save as config.ini within this directory and then execute model.
#==========================================CONFIG FILE======================================================