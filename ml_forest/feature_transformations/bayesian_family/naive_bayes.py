from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from ml_forest.core.elements.ftrans_base import SklearnClassifier


class BNB(SklearnClassifier):
    def __init__(self, binarize=0.00001, predict_proba=False):
        super(BNB, self).__init__(model_type=BernoulliNB, rise=1)

        # get essentials
        essential_keys = {"predict_proba", "binarize"}

        self.__essentials = {}
        for key in essential_keys:
            self.__essentials[key] = locals()[key]


class MNB(SklearnClassifier):
    def __init__(self, predict_proba=False):
        super(MNB, self).__init__(model_type=MultinomialNB, rise=1)

        # get essentials
        essential_keys = {"predict_proba"}

        self.__essentials = {}
        for key in essential_keys:
            self.__essentials[key] = locals()[key]
