import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from ml_forest.core.elements.ftrans_base import FTransform


class SimpleNumImpute(FTransform):
    def __init__(self, indicate_impute=False, ignore=False):
        super(SimpleNumImpute, self).__init__(rise=0)
        self.__cols_info = {}
        self._ignore = ignore
        self._indicate_impute = indicate_impute
        self.__essentials = {"indicate_impute": indicate_impute}

    def fit_whole(self, x):
        if len(x.shape) == 1:
            x.reshape(-1, 1)

        lst = []
        for i in range(x.shape[1]):
            val = x[:, i]
            boo = np.isnan(val).astype(int).reshape(-1, 1)
            max_ = np.nanmax(val)
            min_ = np.nanmin(val)

            if min_ >= 0:
                sub = -10
            elif max_ <= 0:
                sub = 10
            else:
                sub = max(np.abs(min_), np.abs(max_)) * 2

            val = np.array(
                [sub if np.isnan(z) else z for z in val]
            ).reshape(-1, 1)

            if self.indicate_impute:
                val = np.concatenate([val, boo], axis=1)
            lst.append(val)

            self.__cols_info[i] = {
                "rng": (min_, max_), "sub": sub
            }

        value = np.concatenate(lst, axis=1)
        return None, value

    def transform(self, fed_test_value):
        if not bool(self.cols_info):
            msg = "The {} object is not fitted yet".format(str(type(self)))
            raise NotFittedError(msg)

        if fed_test_value.shape[1] != len(self.cols_info):
            msg = "It seems that the test values passed doesn't have the same number of columns as " +\
                  "the feature we imputed with this FTransform object"
            raise ValueError(msg)

        lst = []
        for i in range(fed_test_value.shape[1]):
            min_, max_ = self.cols_info[i]["rng"]
            sub = self.cols_info[i]["sub"]
            val = fed_test_value[:, i]
            boo = np.isnan(val).astype(int).reshape(-1, 1)

            if np.logical_or(val < min_, val > max_).any():
                warning_msg = "There are some elements in test feature out of the range of the original feature. " +\
                              "This jeopardizes the strategy to impute with values far from the common range"
                warnings.warn(warning_msg)
                if sub < min_ and (val < sub).any() and self.ignore:
                    warning_msg = "The method is NOT good: We actually found values smaller than the outlier " +\
                                  "in the negative side that we used."
                    warnings.warn(warning_msg)
                elif sub < min_ and (val < sub).any() and not self.ignore:
                    msg = "This FTransform is NOT good: We actually found values smaller than the outlier in the " +\
                          "negative side that we used."
                    raise ValueError(msg)
                elif sub > max_ and (val > sub).any() and self.ignore:
                    warning_msg = "The method is NOT good: We actually found values greater than the outlier " +\
                                  "in the positive side that we used."
                    warnings.warn(warning_msg)
                elif sub > max_ and (val > sub).any() and not self.ignore:
                    msg = "The FTransform is NOT good: We actually found values greater than the outlier in the " +\
                          "positive side that we used."
                    raise ValueError(msg)
            val = np.array(
                [sub if np.isnan(z) else z for z in val]
            ).reshape(-1, 1)
            if self.indicate_impute:
                val = np.concatenate([val, boo], axis=1)
            lst.append(val)

        value = np.concatenate(lst, axis=1)
        return value

    @property
    def cols_info(self):
        return self.__cols_info

    @property
    def ignore(self):
        return self._ignore

    @property
    def indicate_impute(self):
        return self._indicate_impute


class RandomImpute(FTransform):
    def __init__(self):
        super(RandomImpute, self).__init__(rise=0)
        self.__essentials = {}
        self.candidates = None

    def fit_whole(self, x):
        if len(x.shape) > 2:
            raise ValueError("Not good for high dimension features.")
        elif len(x.shape) == 2 and x.shape[1] != 1:
            raise ValueError("Only good for a single feature for now.")

        fval = pd.Series(x.ravel())
        candidates = list(fval.loc[np.logical_not(fval.isnull())])
        self.candidates = candidates

        fval = self.transform(fval)

        return None, fval

    def transform(self, fed_test_value):
        if not isinstance(fed_test_value, pd.Series):
            fed_test_value = pd.Series(fed_test_value.ravel())
        n = fed_test_value.loc[fed_test_value.isnull()].shape[0]

        imputed_values = fed_test_value.copy()
        imputed_values.loc[imputed_values.isnull()] = np.random.choice(self.candidates, size=n)
        imputed_values = np.array(imputed_values).reshape(-1, 1)

        return imputed_values


class MeanImpute(FTransform):
    def __init__(self):
        super(MeanImpute, self).__init__(rise=0)
        self.__essentials = {}
        self.mean = None

    def fit_whole(self, x):
        if len(x.shape) > 2:
            raise ValueError("Not good for high dimension features.")
        elif len(x.shape) == 2 and x.shape[1] != 1:
            raise ValueError("Only good for a single feature for now.")

        fval = pd.Series(x.ravel())
        self.mean = fval.mean()

        fval = self.transform(fval)

        return None, fval

    def transform(self, fed_test_value):
        if not isinstance(fed_test_value, pd.Series):
            fed_test_value = pd.Series(fed_test_value.ravel())

        imputed_values = fed_test_value.copy()
        imputed_values.loc[imputed_values.isnull()] = self.mean
        imputed_values = np.array(imputed_values).reshape(-1, 1)

        return imputed_values


if __name__ == "__main__":
    print(
        SimpleNumImpute()
    )
