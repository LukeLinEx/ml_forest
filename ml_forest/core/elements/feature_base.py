from bson.objectid import ObjectId

from ml_forest.core.elements.identity import Base
from ml_forest.core.constructions.db_handler import DbHandler


class Feature(Base):
    def __init__(self, frame, lst_fed, f_transformer, label, values, **kwargs):
        """


        :param frame: ObjectId
        :param lst_fed: list of ObjectId or None
        :param label: ObjectId or None
        :param method: ObjectId
        :param kwargs: saving related. Check the attribute in SaveBase
        """
        if frame and not isinstance(frame, ObjectId):
            raise TypeError("The parameter frame should be a obj_id")
        if f_transformer and not isinstance(f_transformer, ObjectId):
            raise TypeError("The parameter f_transformer should be a obj_id")
        if label and not isinstance(label, ObjectId):
            raise TypeError("The parameter label should be a obj_id")
        if lst_fed:
            for f in lst_fed:
                if not isinstance(f, ObjectId):
                    raise TypeError("The parameter lst_fed should consist of obj_id")

        super(Feature, self).__init__(**kwargs)

        self.__values = values

        self.__stage = None

        self.__essentials = {
            'frame': frame,
            'lst_fed': lst_fed,
            'method': f_transformer,
            'label': label
            }

        if type(self) == Feature:
            self.save_db_file()

    @property
    def values(self):
        return self.__values.copy()
