from bson.objectid import ObjectId
from ml_forest.core.elements.identity import Base
from ml_forest.core.constructions.db_handler import DbHandler


class Label(Base):
    def __init__(self, frame, transformer, raw_y, values, db=None, filepaths=None):
        """

        :param frame: ObjectId.
        :param transformer: ObjecId or None
        :param raw_y: ObjectId or None
        :param values: numpy.ndarray. The actual value of the label
        :param db:
        :param filepaths:
        :return:
        """
        if frame and not isinstance(frame, ObjectId):
            raise TypeError("The parameter frame should be a obj_id")
        if transformer and not isinstance(transformer, ObjectId):
            raise TypeError("The parameter transformer should be a obj_id")
        if raw_y and not isinstance(raw_y, ObjectId):
            raise TypeError("The parameter raw_y should be a obj_id")

        super(Label, self).__init__(db, filepaths)

        self.__values = values
        self.__essentials = {
            'transformer': transformer,
            'frame': frame,
            'raw_y': raw_y
        }

        if type(self) == Label:
            if self.db:
                dh = DbHandler()
                obj_id = dh.init_doc(self)
                self.obj_id = obj_id
            # TODO: saving the obj into file
            if self.filepaths:
                raise NotImplementedError("Implement the saving objects")

    @property
    def values(self):
        return self.__values.copy()
