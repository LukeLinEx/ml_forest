from copy import deepcopy
from ml_forest.core.elements.identity import Base

__author__ = 'LukeLin'


class FTransform(Base):
    def __init__(self, rise, tuning=False):
        """

        :param rise: int, indicates how many stages (see feature_base) to rise:
                          unsupervised learning: 0
                          supervised learning: 1
                          supervised learning with tuning: 1
        :param tuning: boolean, if tuning is happening in the training folds

        rise and tuning are fixed for a FTransform class, so they are not saved in essentials since type is
        """
        super(FTransform, self).__init__()
        if not isinstance(rise, int):
            raise TypeError('The {} method has a non-integer rise, need to be updated'.format(str(type(self))))
        self.__tuning = tuning
        self.__rise = rise
        self.__essentials = {}
        self.__models = None

    @property
    def rise(self):
        return self.__rise

    @property
    def models(self):
        return deepcopy(self.__models)

    def record_models(self, model_collection):
        if self.__models:
            raise ValueError("The attribute models is non-empty. This method does not allow for updating.")

        if not isinstance(model_collection, dict):
            raise TypeError("The parameter model_collection should be a dictionary.")
        self.__models = model_collection

    @property
    def tuning(self):
        return self.__tuning

    @staticmethod
    def decide_element():
        return "FTransform"


class SklearnModel(FTransform):
    def __init__(self, model_type, **kwargs):
        super(SklearnModel, self).__init__(**kwargs)
        self.__model_type = model_type

        self.__essentials = {}

    def fit_singleton(self, x, y, new_x):
        model = self.__model_type()
        for key in model.get_params():
            if key in self.__essentials:
                model.set_params(**{key: self.essentials[key]})
        model.fit(x, y)

        return model, model.predict(new_x)

    def transform_singleton(self, model, new_x):
        if "predict_proba" in self.essentials and self.essentials["predict_proba"]:
            return model.predict_proba(new_x)
        else:
            return model.predict(new_x)
