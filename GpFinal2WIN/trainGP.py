from tree import Tree
from ExpressionGenerator import GenMember
import math
from random import random
from convertToInfix import ToInfixParser
from convertToPrefix import ToPrefixParser
# import matplotlib.pyplot as plt
from data import Data
import copy
import timeit


def train_gp(data_set='dataset2.txt', gen_depth=3, population_size=500, max_iteration=2,
             selection_type="tournament", tournament_size=50, cross_over_rate=0.9, mutation_rate=0.1, thresh=0.5):
    """
    Function to train the genetic program using the training dataset, based on user defined parameters.
    :param data_set: data set to be read into the program
    :param gen_depth: depth of the original population
    :param population_size: maximum popululation size
    :param max_iteration: stopping criteria if no solution is found within a reasonable iteration limit
    :param cross_over_rate: frequency of crossover expressed as a value between [0,1]
    :param selection_type: type of selection -> choose between tournament and select best 2 each time
    :param tournament_size: number of individuals to be selected for tournament in population
    :param mutation_rate: frequency of mutation expressed as a value between [0,1]
    :param thresh: testing threshold value to print out the parameters being used.
    :return: optimal expression found through training.
    """
    import sys
    start_time = timeit.default_timer()
    timer = list()
    loop_break = False
    to_pref = ToPrefixParser()
    tree = Tree()
    x_val = list()
    y_val = list()

    d = Data(data_set)
    read = d.read_data()
    data = read[0]
    labels = read[1]

    tsp = d.train_test_split_ds(data, labels)
    x_train = tsp[0]
    y_train = tsp[1]
    x_test = tsp[2]
    y_test = tsp[3]

    sys.stdout.write("###########parameters########### \n")
    sys.stdout.write("dataset: {} \n".format(data_set))
    sys.stdout.write("Generation depth: {} \n".format(gen_depth))
    sys.stdout.write("Population Size : {} \n".format(population_size))
    sys.stdout.write("Maximum Iterations : {} \n".format(max_iteration))
    sys.stdout.write("Selection Type : {} \n".format(selection_type))
    sys.stdout.write("tournament size : {} \n".format(tournament_size))
    sys.stdout.write("Crossover rate : {} \n".format(cross_over_rate))
    sys.stdout.write("Mutation Rate : {} \n".format(mutation_rate))
    sys.stdout.write("Testing Threshold : {} \n".format(thresh))
    sys.stdout.write("################################ \n")

    current_population = GenMember()
    if population_size < 3 and selection_type == "tournament":
        sys.stderr.write("Population size smaller than 3 members \n")
        sys.stderr.write("Population minimum size of 3.\n ")
        population_size = 3
        # print("population size now: ", population_size)

    if selection_type == "tournament" and tournament_size > population_size:
        sys.stderr.write("Population size smaller than tournament size \n")
        sys.stderr.write("reverting back to default.\n ")
        tournament_size = math.ceil(population_size * 0.4)
        # print("tournament size: ", tournament_size)
        # sys.exit()
    if cross_over_rate < 0:
        sys.stderr.write("Crossover rate must be between 0 and 1\n")
        sys.stderr.write("Crossover disabled  \n")
        cross_over_rate = 0
    if cross_over_rate > 1:
        sys.stderr.write("Crossover rate must be between 0 and 1\n")
        sys.stderr.write("Crossover enabled \n")
        cross_over_rate = 1

    if mutation_rate < 0:
        sys.stderr.write("Mutation rate must be between 0 and 1\n")
        sys.stderr.write("Mutation disabled  \n")
        cross_over_rate = 0
    if mutation_rate > 1:
        sys.stderr.write("Mutation rate must be between 0 and 1\n")
        sys.stderr.write("Mutation enabled \n")
        cross_over_rate = 1

    population = current_population.get_valid_expressions(gen_depth, population_size)

    x = 1

    while x <= max_iteration:
        elapse = timeit.default_timer() - start_time
        timer.append(elapse)
        # print("elapsed time: ", elapse)

        if x == 1:
            population_fitness = current_population.get_fitness(population, x_train, y_train, thresh)

        for index in range(len(population_fitness)):
            if population_fitness[index] <= 120:
                print("#########################################################################")
                print(True)

                print("Iteration: ", x)
                print(" Training fitness index:", population_fitness.index(population_fitness[index]))
                print(" Training fitness: ", population_fitness[index])
                print()
                print(population[index])
                loop_break = True

            if loop_break is True:
                return population[index], x_test, y_test, y_val, timer

        if x % 10 == 0:
            x_val.append(x)
            abs_list = [abs(f) for f in population_fitness]
            min_val = min(abs_list)
            sys.stdout.flush()
            y_val.append(min_val)
            elapse = timeit.default_timer() - start_time
            timer.append(elapse)

        if x == max_iteration:
            print("max iteration met")
            # print("fitness: ", get_fitness)
            abs_list = [abs(fit) for fit in population_fitness]
            min_val = min(abs_list)
            print("best fitness: ", min_val)
            index = abs_list.index(min_val)
            print("index: ", index)
            # print("population: ", population)
            print("equation: ", population[index])
            acc = 1 - (min_val / len(x_train))
            print("acc: ", round(acc, 2) * 100, "%")
            # plt.figure()
            # plt.plot(x_val, y_val, "b", label="fitness")
            # plt.xlabel("iteration")
            # plt.ylabel("fitness")
            # plt.legend(loc="best")
            # plt.show()
            # print(timer)
            return population[index], x_test, y_test, y_val, timer
        if selection_type == 'tournament':
            select_parents = current_population.tournament_selection(population, population_fitness, tournament_size)
        elif selection_type == 'select_best':
            select_parents = current_population.select_best_parents(population, population_fitness)

        # print("parents selected", select_parents)
        split_parents = to_pref.split_parents(select_parents)
        # print("split parents:")
        # print(split_parents)
        get_prefix_parents = to_pref.get_prefix_notation(split_parents)
        # print("prefix notation: ")
        # print("parent prefix: ", get_prefix_parents)
        #     #
        #     # print()
        #     # print("parent trees")
        parent_tree1 = get_prefix_parents[0]
        parent_tree2 = get_prefix_parents[1]
        # parent_tree1_fitness = get_prefix_parents[0][1]
        # parent_tree2_fitness = get_prefix_parents[1][1]

        #     # print("here")
        #     # print(parent_tree1_fitness)
        #     # print(parent_tree2_fitness)

        #     # print("p1 prefix:",parent_tree1)
        #     # print("p2 prefix:",parent_tree2)

        #     # print("making trees!")
        make_parent_tree_one = tree.make_tree(parent_tree1[0])
        make_parent_tree_two = tree.make_tree(parent_tree2[0])

        #     # print("Printing trees")
        #     # print("Tree one")
        #     # show_parent_tree_one = tree.print_full_tree(make_parent_tree_one[0])
        #     # print("parent 1")
        #     # print(show_parent_tree_one)
        # show_parent_tree_one_nodes = tree.print_full_tree(make_parent_tree_one[1])
        #     # print(show_parent_tree_one_nodes)
        #     # print("Tree two")
        #     # show_parent_tree_two = tree.print_full_tree(make_parent_tree_two[0])
        #     # print()
        #     # print("parent2: ")
        #     # print(show_parent_tree_two)
        # show_parent_tree_two_nodes = tree.print_full_tree(make_parent_tree_two[1])
        #     # print(show_parent_tree_two_nodes)
        # nodes_parent_tree_one = tree.print_full_tree(make_parent_tree_one[2])
        #     # print("parent one nodes: ", nodes_parent_tree_one)
        # nodes_parent_tree_two = tree.print_full_tree(make_parent_tree_two[2])
        #     # print("parent two nodes: ", nodes_parent_tree_two)

        #     # make a copy of the parents
        make_parent_tree_one_clone = copy.deepcopy(make_parent_tree_one)
        #     # show_parent_tree_one_clone = tree.print_full_tree(make_parent_tree_one_clone[0])
        #     # print("here")
        # parent_tree1_fitness_clone = parent_tree1_fitness
        #     # print(parent_tree1_fitness_clone)
        #     # print(show_parent_tree_one_clone)

        make_parent_tree_two_clone = copy.deepcopy(make_parent_tree_two)
        """
        #     # show_parent_tree_two_clone = tree.print_full_tree(make_parent_tree_two_clone[0])
        parent_tree2_fitness_clone = parent_tree2_fitness
        #     # print(parent_tree2_fitness_clone)
        #     # print(show_parent_tree_two_clone)
        """
        """
        nodes_parent_tree_one_clone = tree.print_full_tree(make_parent_tree_one_clone[2])
        # print("parent one nodes: ", nodes_parent_tree_one)
        nodes_parent_tree_two_clone = tree.print_full_tree(make_parent_tree_two_clone[2])
        # print("parent two nodes: ", nodes_parent_tree_two)
        """

        rnd = random()
        #     # print("rnd : ", rnd)
        if rnd <= cross_over_rate:
            #         # print("crossing over")
            select_xover_node_one = tree.select_random_val(make_parent_tree_one_clone[1])
            #         # print("blooop: ",select_xover_node_one)
            select_xover_node_two = tree.select_random_val(make_parent_tree_two_clone[1])

            #         # print("selected xover point 1: ", select_xover_node_one)
            #         # print("selected xover point 2: ", select_xover_node_two)

            random_node_one = tree.find_subtree(make_parent_tree_one_clone[0], make_parent_tree_one_clone[1],
                                                select_xover_node_one)
            random_node_two = tree.find_subtree(make_parent_tree_two_clone[0], make_parent_tree_two_clone[1],
                                                select_xover_node_two)

            #         # print('swapping: ', random_node_one.value, random_node_one.nodenum, " with ", random_node
            # _two.value,random_node_two.nodenum)

            new_trees = tree.swap_nodes(make_parent_tree_one_clone[0], make_parent_tree_two_clone[0],
                                        random_node_one, random_node_two)
        else:
            #         # print("not crossing over")
            new_trees = [make_parent_tree_one_clone[0], make_parent_tree_two_clone[0]]
            #     # print()
        child_one = new_trees[0]
        child_two = new_trees[1]
        #     # print("child one")
        #     # print(child_one)
        #     # print()
        #     # print("building child two")
        #     # print(child_two)

        child_one_list_node = list(tree.make_list_nodes(child_one))
        child_two_list_node = list(tree.make_list_nodes(child_two))
        child_two_list_node = tree.get_child_two(child_one_list_node, child_two_list_node)

        #     # print("child one nodes: ", child_one_list_node)
        #     # print()
        #     # print("child two nodes: ", child_two_list_node)

        #     # print("mutating nodes: ")
        rnd = random()
        if rnd <= mutation_rate:
            #         # print("mutating nodes: ")
            node_to_mutate_one = tree.select_random_val(child_one_list_node)
            #         # print("node to mutate one: ",node_to_mutate_one)
            #         # print()
            node_to_mutate_two = tree.select_random_val(child_two_list_node)
            #         # print("node to mutate two: ",node_to_mutate_two)
            #         # print()

            new_child_one = tree.mutate_node(child_one, child_one_list_node, node_to_mutate_one[2])
            #         # print(new_child_one[0])
            #         #
            new_child_two = tree.mutate_node(child_two, child_two_list_node, node_to_mutate_two[2])
            #         # print(new_child_two[0])

        else:
            #         #
            #         # print("not mutating:")
            new_child_one = tree.build_child(child_one, child_one_list_node)
            new_child_two = tree.build_child(child_two, child_two_list_node)

            #     # print("deconstructing trees")
        p = ToInfixParser()
        #     # print("deconstructing child 1")
        deconstruct_child_one = ToInfixParser.deconstruct_tree(new_child_one[1])
        # print(deconstruct_child_one)

        c1 = p.get_infix_notation(deconstruct_child_one)
        c1 = c1.replace(" ", "")

        # print("child one: ", c1)
        # print("population :", population)

        # population.append(c1)

        # print("deconstructing child 2")
        deconstruct_child_two = ToInfixParser.deconstruct_tree(new_child_two[1])
        # print(deconstruct_child_two)

        c2 = p.get_infix_notation(deconstruct_child_two)
        c2 = c2.replace(" ", "")
        # print("child two:", c2)
        # print("jere")
        # get the fitness of the
        new_fit1 = current_population.get_fitness(c1, x_train, y_train, thresh, child=True)
        # print("child: ", c1)
        # print("fitness: ", new_fit1)
        new_fit2 = current_population.get_fitness(c2, x_train, y_train, thresh, child=True)
        # print("child 2: ", c2)
        # print("fitness: ", new_fit2)

        # print("population fitness:", population_fitness)
        update_population1 = current_population.update_population(population, population_fitness,
                                                                  c1, new_fit1, c2, new_fit2)

        population = update_population1[0]
        # print(" new population: ", population)
        population_fitness = update_population1[1]
        # print(" new population fitness:: ", population_fitness)

        x += 1
#
# def train_gp_verbose(data_set='dataset2.txt', gen_depth=3, population_size=500, max_iteration=2,
#              selection_type="tournament", tournament_size=50, cross_over_rate=0.9, mutation_rate=0.1, thresh=0.5):
#     """
#     Function to train the genetic program using the training dataset, based on user defined parameters.
#     :param data_set: data set to be read into the program
#     :param gen_depth: depth of the original population
#     :param population_size: maximum popululation size
#     :param max_iteration: stopping criteria if no solution is found within a reasonable iteration limit
#     :param cross_over_rate: frequency of crossover expressed as a value between [0,1]
#     :param selection_type: type of selection -> choose between tournament and select best 2 each time
#     :param tournament_size: number of individuals to be selected for tournament in population
#     :param mutation_rate: frequency of mutation expressed as a value between [0,1]
#     :param thresh: testing threshold value to print out the parameters being used.
#     :return: optimal expression found through training.
#     """
#     import sys
#     start_time = timeit.default_timer()
#     timer = list()
#     loop_break = False
#     to_pref = ToPrefixParser()
#     tree = Tree()
#     x_val = list()
#     y_val = list()
#
#     d = Data(data_set)
#     read = d.read_data()
#     data = read[0]
#     labels = read[1]
#
#     tsp = d.train_test_split_ds(data, labels)
#     x_train = tsp[0]
#     y_train = tsp[1]
#     x_test = tsp[2]
#     y_test = tsp[3]
#
#     sys.stdout.write("###########parameters########### \n")
#     sys.stdout.write("dataset: {} \n".format(data_set))
#     sys.stdout.write("Generation depth: {} \n".format(gen_depth))
#     sys.stdout.write("Population Size : {} \n".format(population_size))
#     sys.stdout.write("Maximum Iterations : {} \n".format(max_iteration))
#     sys.stdout.write("Selection Type : {} \n".format(selection_type))
#     sys.stdout.write("tournament size : {} \n".format(tournament_size))
#     sys.stdout.write("Crossover rate : {} \n".format(cross_over_rate))
#     sys.stdout.write("Mutation Rate : {} \n".format(mutation_rate))
#     sys.stdout.write("Testing Threshold : {} \n".format(thresh))
#     sys.stdout.write("################################ \n")
#
#     current_population = GenMember()
#     if population_size < 3 and selection_type == "tournament":
#         sys.stderr.write("Population size smaller than 3 members \n")
#         sys.stderr.write("Population minimum size of 3.\n ")
#         population_size = 3
#         # print("population size now: ", population_size)
#
#     if selection_type == "tournament" and tournament_size > population_size:
#         sys.stderr.write("Population size smaller than tournament size \n")
#         sys.stderr.write("reverting back to default.\n ")
#         tournament_size = math.ceil(population_size * 0.4)
#         # print("tournament size: ", tournament_size)
#         # sys.exit()
#     if cross_over_rate < 0:
#         sys.stderr.write("Crossover rate must be between 0 and 1\n")
#         sys.stderr.write("Crossover disabled  \n")
#         cross_over_rate = 0
#     if cross_over_rate > 1:
#         sys.stderr.write("Crossover rate must be between 0 and 1\n")
#         sys.stderr.write("Crossover enabled \n")
#         cross_over_rate = 1
#
#     if mutation_rate < 0:
#         sys.stderr.write("Mutation rate must be between 0 and 1\n")
#         sys.stderr.write("Mutation disabled  \n")
#         cross_over_rate = 0
#     if mutation_rate > 1:
#         sys.stderr.write("Mutation rate must be between 0 and 1\n")
#         sys.stderr.write("Mutation enabled \n")
#         cross_over_rate = 1
#
#     population = current_population.get_valid_expressions(gen_depth, population_size)
#
#     x = 1
#
#     while x <= max_iteration:
#         elapse = timeit.default_timer() - start_time
#         timer.append(elapse)
#         # print("elapsed time: ", elapse)
#
#         if x == 1:
#             population_fitness = current_population.get_fitness(population, x_train, y_train, thresh)
#
#         for index in range(len(population_fitness)):
#             if population_fitness[index] <= 120:
#                 print("#########################################################################")
#                 print(True)
#
#                 print("Iteration: ", x)
#                 print(" Training fitness index:", population_fitness.index(population_fitness[index]))
#                 print(" Training fitness: ", population_fitness[index])
#                 print()
#                 print(population[index])
#                 loop_break = True
#
#             if loop_break is True:
#                 return population[index], x_test, y_test, y_val, timer
#
#         if x % 10 == 0:
#             x_val.append(x)
#             abs_list = [abs(f) for f in population_fitness]
#             min_val = min(abs_list)
#             sys.stdout.flush()
#             y_val.append(min_val)
#             elapse = timeit.default_timer() - start_time
#             timer.append(elapse)
#
#         if x == max_iteration:
#             print("max iteration met")
#             # print("fitness: ", get_fitness)
#             abs_list = [abs(fit) for fit in population_fitness]
#             min_val = min(abs_list)
#             print("best fitness: ", min_val)
#             index = abs_list.index(min_val)
#             print("index: ", index)
#             # print("population: ", population)
#             print("equation: ", population[index])
#             acc = 1 - (min_val / len(x_train))
#             print("acc: ", round(acc, 2) * 100, "%")
#             # plt.figure()
#             # plt.plot(x_val, y_val, "b", label="fitness")
#             # plt.xlabel("iteration")
#             # plt.ylabel("fitness")
#             # plt.legend(loc="best")
#             # plt.show()
#             # print(timer)
#             return population[index], x_test, y_test, y_val, timer
#         if selection_type == 'tournament':
#             select_parents = current_population.tournament_selection(population, population_fitness, tournament_size)
#         elif selection_type == 'select_best':
#             select_parents = current_population.select_best_parents(population, population_fitness)
#
#         # print("parents selected", select_parents)
#         split_parents = to_pref.split_parents(select_parents)
#         # print("split parents:")
#         # print(split_parents)
#         get_prefix_parents = to_pref.get_prefix_notation(split_parents)
#         # print("prefix notation: ")
#         # print("parent prefix: ", get_prefix_parents)
#         #     #
#         #     # print()
#         #     # print("parent trees")
#         parent_tree1 = get_prefix_parents[0]
#         parent_tree2 = get_prefix_parents[1]
#         # parent_tree1_fitness = get_prefix_parents[0][1]
#         # parent_tree2_fitness = get_prefix_parents[1][1]
#
#         #     # print("here")
#         #     # print(parent_tree1_fitness)
#         #     # print(parent_tree2_fitness)
#
#         #     # print("p1 prefix:",parent_tree1)
#         #     # print("p2 prefix:",parent_tree2)
#
#         #     # print("making trees!")
#         make_parent_tree_one = tree.make_tree(parent_tree1[0])
#         make_parent_tree_two = tree.make_tree(parent_tree2[0])
#
#         #     # print("Printing trees")
#         #     # print("Tree one")
#         #     # show_parent_tree_one = tree.print_full_tree(make_parent_tree_one[0])
#         #     # print("parent 1")
#         #     # print(show_parent_tree_one)
#         # show_parent_tree_one_nodes = tree.print_full_tree(make_parent_tree_one[1])
#         #     # print(show_parent_tree_one_nodes)
#         #     # print("Tree two")
#         #     # show_parent_tree_two = tree.print_full_tree(make_parent_tree_two[0])
#         #     # print()
#         #     # print("parent2: ")
#         #     # print(show_parent_tree_two)
#         # show_parent_tree_two_nodes = tree.print_full_tree(make_parent_tree_two[1])
#         #     # print(show_parent_tree_two_nodes)
#         # nodes_parent_tree_one = tree.print_full_tree(make_parent_tree_one[2])
#         #     # print("parent one nodes: ", nodes_parent_tree_one)
#         # nodes_parent_tree_two = tree.print_full_tree(make_parent_tree_two[2])
#         #     # print("parent two nodes: ", nodes_parent_tree_two)
#
#         #     # make a copy of the parents
#         make_parent_tree_one_clone = copy.deepcopy(make_parent_tree_one)
#         #     # show_parent_tree_one_clone = tree.print_full_tree(make_parent_tree_one_clone[0])
#         #     # print("here")
#         # parent_tree1_fitness_clone = parent_tree1_fitness
#         #     # print(parent_tree1_fitness_clone)
#         #     # print(show_parent_tree_one_clone)
#
#         make_parent_tree_two_clone = copy.deepcopy(make_parent_tree_two)
#         """
#         #     # show_parent_tree_two_clone = tree.print_full_tree(make_parent_tree_two_clone[0])
#         parent_tree2_fitness_clone = parent_tree2_fitness
#         #     # print(parent_tree2_fitness_clone)
#         #     # print(show_parent_tree_two_clone)
#         """
#         """
#         nodes_parent_tree_one_clone = tree.print_full_tree(make_parent_tree_one_clone[2])
#         # print("parent one nodes: ", nodes_parent_tree_one)
#         nodes_parent_tree_two_clone = tree.print_full_tree(make_parent_tree_two_clone[2])
#         # print("parent two nodes: ", nodes_parent_tree_two)
#         """
#
#         rnd = random()
#         #     # print("rnd : ", rnd)
#         if rnd <= cross_over_rate:
#             #         # print("crossing over")
#             select_xover_node_one = tree.select_random_val(make_parent_tree_one_clone[1])
#             #         # print("blooop: ",select_xover_node_one)
#             select_xover_node_two = tree.select_random_val(make_parent_tree_two_clone[1])
#
#             #         # print("selected xover point 1: ", select_xover_node_one)
#             #         # print("selected xover point 2: ", select_xover_node_two)
#
#             random_node_one = tree.find_subtree(make_parent_tree_one_clone[0], make_parent_tree_one_clone[1],
#                                                 select_xover_node_one)
#             random_node_two = tree.find_subtree(make_parent_tree_two_clone[0], make_parent_tree_two_clone[1],
#                                                 select_xover_node_two)
#
#             #         # print('swapping: ', random_node_one.value, random_node_one.nodenum, " with ", random_node
#             # _two.value,random_node_two.nodenum)
#
#             new_trees = tree.swap_nodes(make_parent_tree_one_clone[0], make_parent_tree_two_clone[0],
#                                         random_node_one, random_node_two)
#         else:
#             #         # print("not crossing over")
#             new_trees = [make_parent_tree_one_clone[0], make_parent_tree_two_clone[0]]
#             #     # print()
#         child_one = new_trees[0]
#         child_two = new_trees[1]
#         #     # print("child one")
#         #     # print(child_one)
#         #     # print()
#         #     # print("building child two")
#         #     # print(child_two)
#
#         child_one_list_node = list(tree.make_list_nodes(child_one))
#         child_two_list_node = list(tree.make_list_nodes(child_two))
#         child_two_list_node = tree.get_child_two(child_one_list_node, child_two_list_node)
#
#         #     # print("child one nodes: ", child_one_list_node)
#         #     # print()
#         #     # print("child two nodes: ", child_two_list_node)
#
#         #     # print("mutating nodes: ")
#         rnd = random()
#         if rnd <= mutation_rate:
#             #         # print("mutating nodes: ")
#             node_to_mutate_one = tree.select_random_val(child_one_list_node)
#             #         # print("node to mutate one: ",node_to_mutate_one)
#             #         # print()
#             node_to_mutate_two = tree.select_random_val(child_two_list_node)
#             #         # print("node to mutate two: ",node_to_mutate_two)
#             #         # print()
#
#             new_child_one = tree.mutate_node(child_one, child_one_list_node, node_to_mutate_one[2])
#             #         # print(new_child_one[0])
#             #         #
#             new_child_two = tree.mutate_node(child_two, child_two_list_node, node_to_mutate_two[2])
#             #         # print(new_child_two[0])
#
#         else:
#             #         #
#             #         # print("not mutating:")
#             new_child_one = tree.build_child(child_one, child_one_list_node)
#             new_child_two = tree.build_child(child_two, child_two_list_node)
#
#             #     # print("deconstructing trees")
#         p = ToInfixParser()
#         #     # print("deconstructing child 1")
#         deconstruct_child_one = ToInfixParser.deconstruct_tree(new_child_one[1])
#         # print(deconstruct_child_one)
#
#         c1 = p.get_infix_notation(deconstruct_child_one)
#         c1 = c1.replace(" ", "")
#
#         # print("child one: ", c1)
#         # print("population :", population)
#
#         # population.append(c1)
#
#         # print("deconstructing child 2")
#         deconstruct_child_two = ToInfixParser.deconstruct_tree(new_child_two[1])
#         # print(deconstruct_child_two)
#
#         c2 = p.get_infix_notation(deconstruct_child_two)
#         c2 = c2.replace(" ", "")
#         # print("child two:", c2)
#         # print("jere")
#         # get the fitness of the
#         new_fit1 = current_population.get_fitness(c1, x_train, y_train, thresh, child=True)
#         # print("child: ", c1)
#         # print("fitness: ", new_fit1)
#         new_fit2 = current_population.get_fitness(c2, x_train, y_train, thresh, child=True)
#         # print("child 2: ", c2)
#         # print("fitness: ", new_fit2)
#
#         # print("population fitness:", population_fitness)
#         update_population1 = current_population.update_population(population, population_fitness,
#                                                                   c1, new_fit1, c2, new_fit2)
#
#         population = update_population1[0]
#         # print(" new population: ", population)
#         population_fitness = update_population1[1]
#         # print(" new population fitness:: ", population_fitness)
#
#         x += 1
