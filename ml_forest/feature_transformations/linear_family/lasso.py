from sklearn.linear_model import Lasso
from ml_forest.core.elements.ftrans_base import SklearnModel


class GenerateLasso(SklearnModel):
    def __init__(
            self, alpha=1.0, fit_intercept=True, max_iter=1000, normalize=True, precompute=False, positive=False,
            random_state=None, selection='cyclic', tol=0.0001, warm_start=False
    ):
        super(GenerateLasso, self).__init__(model_type=Lasso, rise=1)

        # get essentials
        essential_keys = {
            "normalize", "warm_start", "selection", "fit_intercept", "positive",
            "max_iter", "precompute", "random_state", "tol", "alpha"
        }
        self.__essentials = {}
        for key in essential_keys:
            self.__essentials[key] = locals()[key]




if __name__ == "__main__":
    lasso = GenerateLasso()
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)

    # mod, v1 = lasso.fit_singleton(X, y, X)
    # v2 = lasso.transform_singleton(mod, X)
    # print((v1 == v2).all())
