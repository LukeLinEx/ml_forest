import warnings
import numpy as np
from bson.objectid import ObjectId

from ml_forest.core.elements.identity import Base

"""
      Let depth be a parameter for stack; height for feature. Height should be depth + 1
      For ex:
             For stack              For feature
                                    ___________  stage 3
             ---------  layer 1     ___________  stage 2
             ---------  layer 2     ___________  stage 1
             ---------  layer 3     ___________  stage 0

             ^^^^^^^^^              ^^^^^^^^^^^
             depth = 3              height =3 (We don't need height attribute....stage is enough)

        The reason of naming:
            1. depth and layer for stack
            2. height and stage for feature
        is because when creating stack, we do it top down; but bottom up when training. See blow:

                    layer 1      layer 2
                 I  fold[0]      fold[0, 0]
                 I  fold[0]      fold[0, 1]
                 -------------
                 I  fold[1]      fold[1, 0]
                 I  fold[1]      fold[1, 1]
            stage 2        stage 1       stage 0
        However, training starts from the lowest/deepest layer
"""


class Feature(Base):
    def __init__(self, frame, lst_fed, f_transform, label, values):
        """

        :param frame: ObjectId
        :param lst_fed: list of ObjectId or None
        :param label: ObjectId or None
        :param f_transform: ObjectId
        """
        if frame and not isinstance(frame, ObjectId):
            raise TypeError("The parameter frame should be a obj_id")
        if f_transform and not isinstance(f_transform, ObjectId):
            raise TypeError("The parameter f_transformer should be a obj_id")
        if label and not isinstance(label, ObjectId):
            raise TypeError("The parameter label should be a obj_id")
        if lst_fed:
            for f in lst_fed:
                if not isinstance(f, ObjectId):
                    raise TypeError("The parameter lst_fed should consist of obj_id")

        super(Feature, self).__init__()

        # TODO: ref: stacking_node.FNode.get_docs_match_the_fnode
        if values is not None and not isinstance(values, np.ndarray):
            warnings.warn("A Feature whose values are not np.ndarray is created. Do you really want that?")

        self.__values = values
        self.__stage = None
        self.__essentials = {
            'frame': frame,
            'lst_fed': lst_fed,
            'f_transform': f_transform,
            'label': label
            }

    @property
    def values(self):
        return self.__values.copy()

    @staticmethod
    def decide_element():
        return "Feature"

    @property
    def stage(self):
        return self.__stage

    @stage.setter
    def stage(self, val):
        if self.__stage is None:
            self.__stage = val
        elif self.__stage == val:
            pass
        else:
            raise ValueError("The method doesn't support update stage")
