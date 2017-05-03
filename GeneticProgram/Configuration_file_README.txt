#==========================================CONFIG FILE======================================================
#How to use:
#   Please enter the configuration details of how you would like to construct the Genetic Program
#   the structure is as follows:

#   :param cross_over_rate: value between 0 and 1, determines the frequency of crossover in the GP
#   :param population_size: The number of members in the population 
#   :param data_set: data set to train and test the model on
#   :param max_iteration: initial termination criteria if no good solution found within n iterations
#   :param selection_type: either tournament or select_best selection. If using tournament selection, PLEASE ENSURE the tournament size is smaller than the population size. 
#   :param mutation_rate: value between 0 and 1, determines the frequency of mutation in the GP
#   :param population_size: The number of members in the population 
#   :param generation_depth: max depth of starting population - this will change as the model learns and the expressions become larger. 
#   :param classifier_threshold: value to which a company will fall into one class or another. 
#   :param debug_mode: print out all the stages of the model, to understand how the process works. 
#   :param num_hits_thresh: minimum number of errors such that the model can stop learning.  

Example:

[DEFAULT_GP]
cross_over_rate = 0.1
population_size = 500
data_set = ./dataset2.txt
max_iteration = 1000
selection_type = tournament
tournament_size = 12
mutation_rate = 0.9
generation_depth = 4
classifier_threshold = 0.5
debug_mode = True
num_hits_thresh = 130


save as configGP.ini within this directory and then execute model.
#==========================================CONFIG FILE======================================================