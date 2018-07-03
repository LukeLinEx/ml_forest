from sklearn.linear_model import LogisticRegression
from ml_forest.core.elements.ftrans_base import SklearnModel


class GenerateLogistic(SklearnModel):
    def __init__(self, C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100,
                 multi_class='ovr', n_jobs=1, penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                 verbose=0, warm_start=False, predict_proba=False):

        super(GenerateLogistic, self).__init__(model_type=LogisticRegression, rise=1)

        essential_keys = {'C', 'class_weight', 'dual', 'fit_intercept', 'intercept_scaling', 'max_iter',
                          'multi_class', 'n_jobs', 'penalty', 'random_state', 'solver', 'tol', 'warm_start', 'predict_proba'}

        self.__essentials = {}
        for key in essential_keys:
            self.__essentials[key] = locals()[key]
