import numpy as np
import pandas as pd
import warnings
from sklearn.exceptions import NotFittedError

from ml_forest.core.elements.ftrans_base import FTransform


def any_na_in_1darray(x):
    if not isinstance(x, np.ndarray):
        raise TypeError("This function is designed for np.Ndarray")
    if not len(x.shape) == 1:
        raise ValueError("This function is designed for one dimensional array")

    return pd.Series(x).isnull().any()


class SimpleDummy(FTransform):
    def __init__(self):
        super(SimpleDummy, self).__init__(rise=0)
        self.__essentials = {}
        self._col_encoded = None
        self._all_classes = None

    def fit_whole(self, x):
        if len(x.shape) == 2 and x.shape[1] == 1:
            x = x.reshape(-1, )
        elif x.shape == 1:
            pass
        else:
            raise NotImplementedError("Currently not able to dummify multiple columns")

        if self._col_encoded is not None or self._all_classes is not None:
            raise ValueError("This method object has been fitted before; it's not designed with retrained")

        tmp = pd.get_dummies(x)
        if any_na_in_1darray(x):
            col_encoded = list(tmp.columns)
            self._all_classes = col_encoded + ["This is a missing value"]
            self._col_encoded = col_encoded
        else:
            all_classes = list(tmp.columns)
            self._col_encoded = all_classes[:-1]
            self._all_classes = all_classes

        value = tmp[self._col_encoded]
        value = value.values

        return None, value

    def transform(self, fed_test_value):
        if not bool(self._col_encoded):
            raise NotFittedError("The {} object is not fitted yet".format(str(type(self))))

        if len(fed_test_value.shape) == 2 and fed_test_value.shape[1] == 1:
            fed_test_value = fed_test_value.reshape(-1, )
        elif len(fed_test_value.shape) == 1:
            pass
        else:
            raise NotImplementedError("The test data don't have the same shape as in the training data")

        tmp = pd.Series(fed_test_value)

        tmp.loc[tmp.isnull()] = "This is a missing value"
        unseen = tmp.apply(lambda x: x not in self._all_classes)
        if np.sum(unseen) > 0:
            warnings.warn("Warning: some class in test data hasn't been seen in the training data")

        value = (fed_test_value.reshape(-1, 1) == np.array(self._col_encoded).reshape(1, -1)).astype(float)

        value[unseen, :] = np.nan

        return value

    @property
    def col_encoded(self):
        return self._col_encoded

    @property
    def all_classes(self):
        return self._all_classes
