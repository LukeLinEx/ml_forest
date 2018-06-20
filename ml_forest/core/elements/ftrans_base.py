from copy import deepcopy
from ml_forest.core.elements.identity import Base

__author__ = 'LukeLin'


class FTransform(Base):
    def __init__(self, rise, tuning=False, **kwargs):
        """

        :param rise: int, indicates how many stages (see feature_base) to rise:
                          unsupervised learning: 0
                          supervised learning: 1
                          supervised learning with tuning: 1
        :param tuning: boolean, if tuning is happening in the training folds

        rise and tuning are fixed for a FTransform class, so they are not saved in essentials since type is
        :param kwargs:
        """
        super(FTransform, self).__init__(**kwargs)
        if not isinstance(rise, int):
            raise TypeError('The {} method has a non-integer rise, need to be updated'.format(str(type(self))))
        self.__tuning = tuning
        self.__rise = rise
        self.__essentials = {}
        self.__models = {}

    @property
    def rise(self):
        return self.__rise

    @property
    def models(self):
        return deepcopy(self.__models)

    @property
    def tuning(self):
        return self.__tuning
