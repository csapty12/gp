import numpy as np
from sklearn.model_selection import train_test_split


class Data(object):
    """
    Class to read data, and  manipulate the data such to be shuffled, as well as split th
    """

    def __init__(self, text_file):
        self.text_file = text_file

    def read_data(self):
        """
        Function to load in the text file. Function splits the data into two sets.
        set 1: company data
        set 2: company data labels - either a 0 or 1.
        :return: tuple - (company data, company class)
        """
        from numpy import loadtxt
        cfd = loadtxt(self.text_file)  # read in the data

        class_labels_cfd = cfd[:, -1]  # get the classification categories - [0,1].
        class_labels_cfd = [int(x) for x in class_labels_cfd]
        class_labels_cfd = np.asarray(class_labels_cfd, dtype=int)

        data_cfd = cfd[:, 0:-1]
        return data_cfd, class_labels_cfd

    def train_test_split_ds(self, data, label):
        x = data
        y = label
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)
        return x_train, y_train, x_test, y_test
