from datetime import datetime
from copy import deepcopy
from ml_forest.core.utils.connect_mongo import connect_collection


class DbHandler(object):
    def __init__(self):
        pass

    def mongo_doc_generator(self, doc):
        """
        Here we need to change the type of the values that can't be encoded to mongodb into string
        :param docs: dictionary

            ::types not need to be changed:
            1. float/int/string
            2. datetime
            3. bson.objectid.ObjectId

            ::types need to be changed:
            1. python type object
            2. callable (no need to put in types_not_encodible down there)

        :return: dictionary
        """
        types_not_encodible = [type]

        def encode_switch(obj):
            if type(obj) in types_not_encodible:
                return str(obj)
            elif callable(obj):
                return "{}.{}".format(obj.__module__, obj.__name__)
            else:
                return obj

        doc = deepcopy(doc)
        resulted_doc = {key: encode_switch(doc[key]) for key in doc}

        return resulted_doc

    def init_doc(self, obj):
        """
        The "essentials" attribute of an obj would be used to identify the obj from the db.

        :param obj:
        :return:
        """
        try:
            obj.essentials
        except AttributeError:
            raise AttributeError("An object to be saved in db is supposed to have the essentials attribute")

        if obj.essentials is None:
            raise AttributeError("An object to be saved in db should not have NoneType as its essentials")

        print("Saving this object into db: {}".format(type(obj)))

        start = datetime.now()
        essen = self.mongo_doc_generator(obj.essentials)
        document = {"essentials": essen, 'datetime': start, 'filepaths': obj.filepaths}

        db_location = obj.db
        element = obj.decide_element()
        target_db = connect_collection(
            db_location["host"], db_location["project"], element
        )
        doc_created = target_db.insert_one(document)
        return doc_created.inserted_id


if __name__ == "__main__":
    import numpy as np
    ary = np.array([1,2,3])
    print(ary.__class__.__bases__)
