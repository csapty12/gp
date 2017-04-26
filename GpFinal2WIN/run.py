from trainGP import train_gp
from data import Data
import sys

import numpy


def gp_full_data(test_dataset):
    d = Data(test_dataset)
    x = d.read_data()
    return x[0], x[1]


def run_gp(data_set, thresh=0.5):
    import math
    accs = list()

    for i in range(2):
        optimal_expression = train_gp(data_set=data_set, gen_depth=3, max_depth=3,
                                      population_size=500, max_iteration=5, selection_type="tournament",
                                      tournament_size=50, cross_over_rate=0.5, mutation_rate=0.99, thresh=thresh)

        opt_exp = optimal_expression[0]
        row = optimal_expression[1]
        label = optimal_expression[2]


        exp = list()
        exp.append(opt_exp)
        optimal_expression = exp
        prediction = list()
        for i in optimal_expression:
            tmp = list()
            for j in row:
                new_exp = i.replace("X1", str(j[0])).replace("X2", str(j[1])).replace("X3", str(j[2])) \
                    .replace("X4", str(j[3])).replace("X5", str(j[4]))
                eva = eval(new_exp)
                if eva >= 0:
                    x = eva
                    tmp.append(x)
                else:
                    y = eva
                    tmp.append(y)
            prediction.append(tmp)

        prob = list()

        for i in prediction:
            for j in i:
                try:
                    sig = 1 / (1 + math.exp(-j))
                except OverflowError:
                    sig = 0
                if sig > thresh:
                    prob.append(1)
                else:
                    prob.append(0)
        print("expression: ", optimal_expression)
        # print("classifications")
        # print(prob)

        trufa = prob == label
        # for i in range(len(prob)):
        #     trufa.append(i == label[i])

        # print("true false array")
        # print(trufa)
        ls = sum(trufa) / len(trufa)
        print("accuracy: ", ls)
        accs.append(ls)
    print("accs")
    print(accs)
    for i in range(len(accs)):
        sys.stdout.write(accs[i])
        if i %10 ==0:
            sys.stdout.write("\n")



if __name__ == "__main__":
    run_gp('dataset2.txt')


# TODO - IMPLEMENT LEVEL CAP - OR AT LEAST HANDLE IT SOMEHOW
