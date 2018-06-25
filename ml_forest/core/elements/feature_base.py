from bson.objectid import ObjectId

from ml_forest.core.elements.identity import Base


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

        self.__values = values

        self.__stage = None

        self.__essentials = {
            'frame': frame,
            'lst_fed': lst_fed,
            'method': f_transform,
            'label': label
            }

    @property
    def values(self):
        return self.__values.copy()

    @staticmethod
    def decide_element():
        return "Feature"
