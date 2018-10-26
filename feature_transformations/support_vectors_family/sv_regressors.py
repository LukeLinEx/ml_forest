from ml_forest.core.elements.ftrans_base import SklearnRegressor
from sklearn.svm import SVR


class GenerateSVR(SklearnRegressor):
    def __init__(self, kernel='rbf', degree=3, gamma='auto',
                 coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=False, max_iter=-1):
        super(GenerateSVR, self).__init__(model_type=SVR, rise=1)

        essential_keys = {
            "max_iter", "cache_size", "shrinking", "epsilon",
            "C", "tol", "coef0", "gamma", "degree", "kernel"}
        self.__essentials = {}
        for key in essential_keys:
            self.__essentials[key] = locals()[key]
