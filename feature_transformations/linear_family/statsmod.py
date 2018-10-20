import pandas as pd
import statsmodels.formula.api as sm
from ml_forest.core.elements.ftrans_base import FTransform


class CookDistance(FTransform):
    def __init__(self):
        super(CookDistance, self).__init__(rise=1)
        self.__essentials = {}

    def fit_singleton(self, x, y, new_x):
        x = x[:,0]
        y = y[:,0]
        data = pd.DataFrame({"x": x, "y": y})

        model = sm.ols(formula='y ~ x', data=data)
        fitted = model.fit()
        influence = fitted.get_influence()
        (c, p) = influence.cooks_distance

        return fitted, c

    def transform(self, fed_test_value):
        msg = "Cook distance cannot be computed without a label." +\
              "Therefore, tansformation on a text dataset cannot be done."
        raise AttributeError(msg)
