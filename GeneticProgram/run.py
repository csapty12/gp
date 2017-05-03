from trainGP import train_gp, train_gp_verbose
from data import Data
import timeit
import os.path
import configparser


def gp_full_data(test_dataset):
    d = Data(test_dataset)
    x = d.read_data()
    return x[0], x[1]


def make_config():
    """
    If a configuration file does not exist, then create a default one with default parameters.
    :return: the configuration file
    """
    config = configparser.ConfigParser()
    config['DEFAULT_GP'] = {
        'data_set': './dataset2.txt',
        'generation_depth': '4',
        'population_size': '500',
        'max_iteration': '1000',
        'selection_type': 'tournament',
        'tournament_size': '40',
        'cross_over_rate': '0.1',
        'mutation_rate': '0.9',
        'classifier_threshold': '0.5',
        'num_hits_thresh': '130',
        'debug_mode': 'False',
        'train_size': '0.8',
        'test_size': '0.2'

    }
    with open('configGP.ini', 'w') as configfile:
        config.write(configfile)


def read_config_file(file):
    """
    function to read the configuration file which takes the parameters and is used to train and test the ANN.
    :param file: the configuration file
    :return: the configuration parameters.
    """
    config = configparser.ConfigParser()
    config.read(file)
    params = list()
    # get the key values for each parameter
    data_set = config['DEFAULT_GP']['data_set']
    generation_depth = config['DEFAULT_GP']['generation_depth']
    population_size = config['DEFAULT_GP']['population_size']
    max_iteration = config['DEFAULT_GP']['max_iteration']
    selection_type = config['DEFAULT_GP']["selection_type"]
    tournament_size = config['DEFAULT_GP']["tournament_size"]
    cross_over_rate = config['DEFAULT_GP']["cross_over_rate"]
    muataion_rate = config['DEFAULT_GP']["mutation_rate"]
    classifier_threshold = config['DEFAULT_GP']["classifier_threshold"]
    num_hit_threshold = config['DEFAULT_GP']["num_hits_thresh"]
    debug_mode = config['DEFAULT_GP']["debug_mode"]
    train_size = config['DEFAULT_GP']["train_size"]
    test_size = config['DEFAULT_GP']["test_size"]

    params.append(data_set)
    params.append(generation_depth)
    params.append(population_size)
    params.append(max_iteration)
    params.append(selection_type)
    params.append(tournament_size)
    params.append(cross_over_rate)
    params.append(muataion_rate)
    params.append(classifier_threshold)
    params.append(debug_mode)
    params.append(num_hit_threshold)
    params.append(train_size)
    params.append(test_size)
    print("parameters from config file being used:")
    print(params)
    return params


def run_gp(config_file):
    data_set = config_file[0]
    init_depth = int(config_file[1])
    pop_size = int(config_file[2])
    max_iter = int(config_file[3])
    sel_type = config_file[4]
    tourn_size = int(config_file[5])
    x_over_rate = float(config_file[6])
    mut_rate = float(config_file[7])
    thresh = float(config_file[8])
    verbose = config_file[9]
    num_hits = int(config_file[10])
    train_size = float(config_file[11])
    test_size = float(config_file[12])
    import math
    accs = list()
    timer = list()
    start_time = timeit.default_timer()

    for i in range(1):
        elapse = timeit.default_timer() - start_time
        timer.append(elapse)
        if verbose == "False":
            optimal_expression = train_gp(data_set=data_set, gen_depth=init_depth,
                                          population_size=pop_size, max_iteration=max_iter, selection_type=sel_type,
                                          tournament_size=tourn_size, cross_over_rate=x_over_rate,
                                          mutation_rate=mut_rate, thresh=thresh, number_hits=num_hits,
                                          training_size=train_size,
                                          testing_size=test_size)
        elif verbose == "True":
            optimal_expression = train_gp_verbose(data_set=data_set, gen_depth=init_depth,
                                                  population_size=pop_size, max_iteration=max_iter,

                                                  selection_type=sel_type,
                                                  tournament_size=tourn_size, cross_over_rate=x_over_rate,
                                                  mutation_rate=mut_rate,
                                                  thresh=thresh, number_hits=num_hits, training_size=train_size,
                                                  testing_size=test_size)

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
    file_path = './configGP.ini'
    if os.path.exists(file_path):
        print("here")
        configuration_file = read_config_file(file_path)
    else:
        print("No config file detected, creating default config.ini file")
        make_config()
        configuration_file = read_config_file(file_path)
    classifier = run_gp(configuration_file)
    draw_roc(classifier[0], classifier[1])
