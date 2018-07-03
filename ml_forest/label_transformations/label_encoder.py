import numpy as np
import pandas as pd
from ml_forest.core.elements.ltrans_base import LTransform


class TrivialEncoder(LTransform):
    def __init__(self, **kwargs):
        """

        :param kwargs:  saving path related
        :return:
        """
        super(TrivialEncoder, self).__init__(**kwargs)

        # attributes characterize the instance
        self._classes=None
        self.__essentials = {}

    def encode_whole(self, fed_y):
        """

        :param fed_y: label before being transformed
        :return: the label transformed; ready for training
        """
        if len(fed_y.shape) > 1:
            fed_y = fed_y.ravel()

        dummy = pd.get_dummies(fed_y)
        self._classes = np.array(dummy.columns)
        encoded_values = np.argmax(dummy.values, axis=1)
        encoded_values = encoded_values.reshape(-1, 1)

        return encoded_values

    def encode_test_label(self, fed_y_value):
        if not isinstance(fed_y_value, np.ndarray):
            raise TypeError("fed_y_value is supposed to be a numpy array")

        encoded_values = np.argmax(
            fed_y_value.reshape(-1, 1)==self._classes, axis=1
        )
        encoded_values = encoded_values.reshape(-1, 1)

        return encoded_values

    def decode(self, new_y):
        return self._classes[new_y].reshape(-1, 1)

    def decode_proba_prediction(self, new_y):
        return pd.DataFrame(new_y, columns=self._classes)
