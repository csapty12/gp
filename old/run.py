from trainGP import train_gp
from data import Data
from sklearn.model_selection import train_test_split
import numpy


def test_gp_full_data(test_dataset):
    d = Data(test_dataset)
    x = d.read_data()
    return x[0], x[1]

def train_test_split_ds(data, label):
    X = data
    y = label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, y_train, X_test, y_test


def run_gp(data_set, thresh = 0.5):
    import math
    accs = list()
    print("hello")
    # for i in range(20):
    #     optimal_expression = train_gp(data_set=data_set, gen_depth=3, max_depth=3,
    #                                   population_size=500, max_iteration=1000, selection_type="tournament",
    #                                   tournament_size=50, cross_over_rate=0.5, mutation_rate=0.99, thresh = thresh )
    #
    #     x = test_gp_full_data(data_set)
    #     row = x[0]
    #     label = x[1]
    #
    #     exp = list()
    #     exp.append(optimal_expression)
    #     optimal_expression = exp
    #     # row = [[0.185841328, 0.229878245, 0.150353322, 2.267962444, 1.72085425],
    #     #        [0.16285377, 0.293619897, 0.148429586, 2.112106101, 1.726711829],
    #     #        [0.149332758, 0.347589881, 0.139985797, 1.689751437, 1.734865801],
    #     #        [0.137193647, 0.416721256, 0.147865432, 2.116532577, 1.761369401],
    #     #        [0.082350665, 0.480389313, 0.174387346, 2.342011704, 1.766493641],
    #     #        [0.159720391, -0.781208802, -0.087774755, 0.333050959,1.899437307]]
    #     # label = [0, 0, 0, 0, 0, 1]
    #     #
    #     prediction = list()
    #     for i in optimal_expression:
    #         tmp = list()
    #         for j in row:
    #             new_exp = i.replace("X1", str(j[0])).replace("X2", str(j[1])).replace("X3", str(j[2])) \
    #                 .replace("X4", str(j[3])).replace("X5", str(j[4]))
    #             eva = eval(new_exp)
    #             if eva >= 0:
    #                 x = eva
    #                 tmp.append(x)
    #             else:
    #                 y = eva
    #                 tmp.append(y)
    #         prediction.append(tmp)
    #
    #     prob = list()
    #
    #     for i in prediction:
    #         for j in i:
    #             try:
    #                 sig = 1 / (1 + math.exp(-j))
    #             except OverflowError:
    #                 sig = 0
    #             if sig > thresh:
    #                 prob.append(1)
    #             else:
    #                 prob.append(0)
    #     print("expression: ", optimal_expression)
    #     # print("classifications")
    #     # print(prob)
    #
    #     trufa = prob == label
    #     # for i in range(len(prob)):
    #     #     trufa.append(i == label[i])
    #
    #     # print("true false array")
    #     # print(trufa)
    #     ls = sum(trufa) / len(trufa)
    #     print("accuracy: ", ls)
    #     accs.append(ls)
    # print("accs")
    # print(accs)


if __name__ == "__main__":
    run_gp('dataset2.txt')


# TODO - IMPLEMENT LEVEL CAP - OR AT LEAST HANDLE IT SOMEHOW
