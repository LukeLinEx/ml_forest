import math
import numpy as np

from ml_forest.core.elements.ltrans_base import LTransform


class LogTransform(LTransform):
    def __init__(self, base=None, shift=0):
        super(LogTransform, self).__init__()
        if base:
            self.__base = base
        else:
            self.__base = math.e

        self.__shift = shift
        self.__essentials = {"base": self.__base, "shift": self.__shift}

    def encode_whole(self, fed_y):
        """

        :param fed_y: label before being transformed
        :return: the label transformed; ready for next step
        """
        b = self.__base
        encoded_values = np.log(fed_y + self.__shift) / np.log(b)
        if len(encoded_values.shape) < 2:
            encoded_values = encoded_values.reshape(-1, 1)

        return encoded_values

    def encode_test_label(self, fed_y_value):
        encoded_values = self.encode_whole(fed_y_value)

        return encoded_values

    def decode(self, new_y):
        tmp = np.log(self.__base) * new_y
        decoded_values = np.exp(tmp) - self.__shift

        return decoded_values


def test():
    b = 3.1
    s = 1

    tmp = np.array([[np.random.normal()*10] for _ in range(80)])
    tmp = tmp - np.min(tmp) + 2

    lt = LogTransform(b, s)
    transformed = lt.encode_whole(tmp)

    print((transformed == np.log(tmp+s)/np.log(b)).all())
    print(((lt.decode(transformed) - tmp) < 0.0000001).all())