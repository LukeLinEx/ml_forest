from ml_forest.core.elements.ftrans_base import SklearnRegressor
from sklearn.linear_model import LinearRegression


class GenerateLinearReg(SklearnRegressor):
    def __init__(self, fit_intercept=True, normalize=False):
        super(GenerateLinearReg, self).__init__(model_type=LinearRegression, rise=1)

        # get essentials
        essential_keys = {"fit_intercept", "normalize"}
        self.__essentials = {}
        for key in essential_keys:
            self.__essentials[key] = locals()[key]
