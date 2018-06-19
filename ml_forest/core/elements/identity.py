from copy import deepcopy


class Base(object):
    def __init__(self, db=None, filepaths=None):
        """
        Base collects the basic infos, db, filepaths, types and obj_id for the ml_forest.core.elements

        :param db: dictionary={"host": host_name, "project":project_name}
                    host_name identifies the address of the db;
                    project_name identifies the project;
        :param filepaths: lst of dictionaries, each dictionary specifies where pkl file is saved.
            Currently supports below:
            [
                {'home': home, 'project':project_name},
                {'bucket': aws_bucket, 'project':project_name}
            ]
        :return:
        """
        if db and not isinstance(db, dict):
            raise TypeError("Currently only support db of the dictionary type")
        if filepaths and not isinstance(filepaths, list):
            raise TypeError("Currently the collection of the file paths has to be of the list type")
        elif filepaths:
            for path in filepaths:
                if not isinstance(path, dict):
                    raise TypeError("Currently the file paths have to be of the dictionary type")

        self.__essentials = {'type': type(self)}

        if db:
            self.__db = db
        else:
            self.__db = None
        if filepaths:
            self.__filepaths = filepaths
        else:
            self.__filepaths = []

        self.__filename = None
        self.__obj_id = None

    @property
    def db(self):
        return deepcopy(self.__db)

    def set_db(self, val):
        assert not bool(self.db), "The set_db method in Base does not allow reseting the db location"
        self.__db = val

    @property
    def filepaths(self):
        return deepcopy(self.__filepaths)

    def set_filepaths(self, val):
        assert not bool(self.filepaths), "The set_filepaths method in Base does not allow reseting the file paths"
        self.__filepaths = val

    @property
    def obj_id(self):
        return self.__obj_id

    @obj_id.setter
    def obj_id(self, val):
        assert not bool(self.obj_id), "The obj_id cannot be reset."
        self.__obj_id = val

    @property
    def essentials(self):
        """
        :param self: The purpose is to collect all the essentials from all the parent classes.
        :param doc: dictionary
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

    @classmethod
    def decide_element(cls):
        return cls.__name__


if __name__ == "__main__":
    b = Base()
    print(b.decide_element())
