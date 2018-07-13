from copy import deepcopy
from ml_forest.core.constructions.db_handler import DbHandler
from ml_forest.core.constructions.io_handler import IOHandler


class Base(object):
    def __init__(self):
        """
        Base collects the basic infos, db, filepaths, types and obj_id for the ml_forest.core.elements
        :return:
        """
        self.__essentials = {'type': type(self)}
        self.__db = None
        self.__filepaths = []
        self.__obj_id = None

    def set_db(self, db):
        """

        :param db: dictionary={"host": host_name, "project":project_name}
                    host_name identifies the address of the db;
                    project_name identifies the project;
        :return:
        """
        if self.db:
            raise AttributeError("The set_db method in Base does not allow reseting the db location")
        if db and not isinstance(db, dict):
            raise TypeError("Currently only support db of the dictionary type")
        self.__db = db

    @property
    def db(self):
        return deepcopy(self.__db)

    def set_filepaths(self, filepaths):
        """

        :param filepaths: lst of dictionaries, each dictionary specifies where pkl file is saved.
            Currently supports below:
            [
                {'home': home, 'project':project_name},
                {'bucket': aws_bucket, 'project':project_name}
            ]
        :return:
        """
        if self.obj_id is None:
            msg = "The object doesn't have an obj_id, which means it's not saved in db yet," +\
                  "so it should not be saved in storage either."
            raise AttributeError(msg)

        if self.filepaths:
            raise AttributeError("The set_filepaths method in Base does not allow reseting the file paths.")
        if filepaths and not isinstance(filepaths, list):
            raise TypeError("Currently the collection of the file paths has to be of the list type")
        elif filepaths:
            for path in filepaths:
                if not isinstance(path, dict):
                    raise TypeError("Currently the file paths have to be of the dictionary type")

        dh = DbHandler()
        dh.update_doc(self, {"filepaths": filepaths})
        self.__filepaths = filepaths

    @property
    def filepaths(self):
        return deepcopy(self.__filepaths)

    @property
    def obj_id(self):
        return self.__obj_id

    @obj_id.setter
    def obj_id(self, val):
        if self.obj_id:
            raise AttributeError("The obj_id cannot be reset.")
        self.__obj_id = val

    @property
    def essentials(self):
        """
        :param self: The purpose is to collect all the essentials from all the parent classes.
        :return:
        """
        _type = type(self)
        doc = {}
        while _type != object:
            tmp = self.__getattribute__("_{}__essentials".format(_type.__name__))
            for key in tmp:
                if key not in doc:
                    doc[key] = tmp[key]
            _type = _type.__bases__[0]

        return doc

    def save_db(self, db):
        """

        :param db: dict
        :return:
        """
        self.set_db(db)
        dh = DbHandler()
        obj_id = dh.init_doc(self)
        self.obj_id = obj_id

    def save_file(self, filepaths):
        """

        :param filepaths: list of dict
        :return:
        """
        self.set_filepaths(filepaths)
        ih = IOHandler()
        ih.save_obj2file(self)

    def save_db_file(self, db, filepaths):
        self.save_db(db)
        self.save_file(filepaths)

    @staticmethod
    def decide_element():
        raise NotImplementedError

if __name__ == "__main__":
    pass
