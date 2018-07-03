from bson.objectid import ObjectId
from ml_forest.core.elements.identity import Base


class Label(Base):
    def __init__(self, frame, l_transform, raw_y, values):
        """

        :param frame: ObjectId.
        :param l_transform: ObjecId or None
        :param raw_y: ObjectId or None
        :param values: numpy.ndarray. The actual value of the label
        :return:
        """
        if frame and not isinstance(frame, ObjectId):
            raise TypeError("The parameter frame should be a obj_id")
        if l_transform and not isinstance(l_transform, ObjectId):
            raise TypeError("The parameter l_transformer should be a obj_id")
        if raw_y and not isinstance(raw_y, ObjectId):
            raise TypeError("The parameter raw_y should be a obj_id")

        super(Label, self).__init__()

        self.__values = values
        self.__essentials = {
            'l_transform': l_transform,
            'frame': frame,
            'raw_y': raw_y
        }

    @property
    def values(self):
        return self.__values.copy()

    @staticmethod
    def decide_element():
        return "Label"
