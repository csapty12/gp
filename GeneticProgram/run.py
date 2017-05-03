from trainGP import train_gp, train_gp_verbose
from data import Data
import timeit


def gp_full_data(test_dataset):
    d = Data(test_dataset)
    x = d.read_data()
    return x[0], x[1]


def run_gp(data_set, thresh=0.5, verbose=False):
    import math
    accs = list()
    timer = list()
    start_time = timeit.default_timer()

    for i in range(1):
        elapse = timeit.default_timer() - start_time
        timer.append(elapse)
        if verbose is False:
            optimal_expression = train_gp(data_set=data_set, gen_depth=4,
                                          population_size=500, max_iteration=2, selection_type="tournament",
                                          tournament_size=40, cross_over_rate=0.99, mutation_rate=0.99, thresh=thresh)
        elif verbose is True:
            optimal_expression = train_gp_verbose(data_set=data_set, gen_depth=4,
                                                  population_size=500, max_iteration=2, selection_type="tournament",
                                                  tournament_size=40, cross_over_rate=0.99, mutation_rate=0.99,
                                                  thresh=thresh)

        opt_exp = optimal_expression[0]
        row = optimal_expression[1]
        label = optimal_expression[2]
        # print("training times")
        training_time = optimal_expression[4]
        # print(training_time)
        training_fitnesses = optimal_expression[3]

        # print(training_fitnesses)
        exp = list()
        exp.append(opt_exp)
        optimal_expression = exp
        prediction = list()
        for i in optimal_expression:
            tmp = list()
            for j in row:

                new_exp = i.replace("X1", str(j[0])).replace("X2", str(j[1])).replace("X3", str(j[2])) \
                    .replace("X4", str(j[3])).replace("X5", str(j[4]))
                try:
                    eva = eval(new_exp)
                except ZeroDivisionError:
                    eva = 0
                if eva >= 0:
                    x = eva
                    tmp.append(x)
                else:
                    y = eva
                    tmp.append(y)
            prediction.append(tmp)

        prob = list()
        err = list()
        for i in prediction:
            for j in i:
                try:
                    sig = 1 / (1 + math.exp(-j))
                except OverflowError:
                    sig = 0
                except ZeroDivisionError:
                    sig = 0
                err.append(sig)
                if sig > thresh:
                    prob.append(1)
                else:
                    prob.append(0)

        # true false array to compare the predicted values against the actual class labels.
        trufa = prob == label
        ls = sum(trufa) / len(trufa)
        # print("accuracy: ", ls)
        accs.append(ls)
    # print("accs")
    # print(accs)

    save_file = open("./out.txt", 'a')
    for i in accs:
        save_file.write(str(i))
        save_file.write(", ")

    save_file.write("\n")
    save_file.close()
    # print("time taken over n runs")
    # print(timer)
    return label, err


def draw_roc(label, err):
    """
    Function to draw the ROC curve for the GP, to determine whether the results have been produced by chance or not.
    :param label:
    :param err:
    :return:
    """
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    actual = label
    predictions = err
    # get the true and false positive rates for the confusion matrix to be used to draw the curve
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
             label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    # delimit the x and y lengths
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == "__main__":
    clasifier = run_gp('dataset2.txt', verbose=True)
    draw_roc(clasifier[0], clasifier[1])



# TODO - IMPLEMENT LEVEL CAP - OR AT LEAST HANDLE IT SOMEHOW
