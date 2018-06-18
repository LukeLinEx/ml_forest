from copy import deepcopy
from ml_forest.core.utils.connect_mongo import connect_collection


class DbHandler(object):
    def __init__(self):
        pass

    @staticmethod
    def collect_essentials(obj):
        """
        :param obj: The purpose is to collect all the essentials from all the parent classes of this obj.
        :param doc: dictionary
        :return:
        """
        _type = type(obj)
        doc = {}
        while _type != object:
            tmp =  obj.__getattribute__("_{}__essentials".format(_type.__name__))
            for key in tmp:
                if key not in doc:
                    doc[key] = tmp[key]
            _type = _type.__bases__[0]

        return doc


if __name__ == "__main__":
    import numpy as np
    ary = np.array([1,2,3])
    print(ary.__class__.__bases__)
