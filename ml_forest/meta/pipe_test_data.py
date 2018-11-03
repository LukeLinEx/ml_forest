import numpy as np
from bson.objectid import ObjectId

from ml_forest.core.elements.identity import Base
from ml_forest.core.elements.feature_base import Feature

from ml_forest.core.constructions.io_handler import IOHandler
from ml_forest.core.constructions.db_handler import DbHandler

from ml_forest.pipeline.nodes.stacking_node import FNode

from ml_forest.meta.test_feature import TestFeature


class PipeTestData(Base):
    def __init__(self, test_data=None, core=None, tag=None, obj_id=None, db=None, filepaths=None):
        """
        Creating the starting point for the test data. We need to select the same columns, and identify them with
        the core.init_features.

        PipeTestData can have label or not. If not, it's not evaluable, we can only predict.

        Through predicting, (the obj_id of) new TestFeature would be added to the PipeTestData.__test_features

        Empty essentials for PipeTestData. A test process is identified by a test data frame, which is not
        part of what we are tracking.

        :param test_data: A pandas DataFrame of test data
        :param core: ml_forest.core.construction.core_init.CoreInit object provides:
                - _column_groups/col_selected: the dictionary of {feature_name: [col_names]}.
                - init_features: the dictionary of {feature_name: feature.obj_id}
                - col_y: the name for the column of the labels. Used for evaluation.
                - init_label.obj_id
        """
        if obj_id is not None:
            pass
        else:
            db = core.db
            filepaths = core.filepaths

            super(PipeTestData, self).__init__()
            self.__essentials = {"core_init": core.obj_id}

            dh = DbHandler()
            self.test_features = {}
            self.__get_init_test_features(core, test_data)
            self.__label = PipeTestData.__get_label(core, test_data)
            if type(self) == PipeTestData:
                self.save_db(db)
                dh.insert_tag(self, {"tag": tag})
                self.save_file(filepaths)
                print(self.obj_id)

    @staticmethod
    def decide_element():
        return "PipeTestData"

    def __get_init_test_features(self, core, test_data):
        for name in core.init_features:
            colnames = core._column_groups[name]
            fval = test_data[colnames].values

            test_f = TestFeature(self.obj_id, fval)
            test_f.save_db_file(core.db, core.filepaths)
            self.test_features[core.init_features[name]] = test_f.obj_id

    def test_feature_exists(self, oid):
        return oid in self.test_features

    def predict(self, feature, update_file=True):
        """
        output the test_feature's values


        :param feature: Feature/FNode/ObjectId that represents the feature
        :return:
        """
        if not isinstance(feature, Feature) and not isinstance(feature, ObjectId) \
                and not isinstance(feature, FNode):
            msg = "the feature has to be of the Feature, the FNode or the ObjectId class."
            raise TypeError(msg)
        if isinstance(feature, Feature) or isinstance(feature, FNode):
            try:
                feature = feature.obj_id
            except AttributeError:
                msg = "It's likely that the Feature you passed hasn't been materialized."
                raise AttributeError(msg)
        oid = feature

        ih = IOHandler()
        dh = DbHandler()

        if self.test_feature_exists(oid):
            # load the obj and return values
            _id = self.test_features[oid]
            test_feature = ih.load_obj_from_file(_id, "TestFeature", self.filepaths)
            return test_feature.values

        else:
            # create the TestFeature and return values
            f_doc = dh.search_by_obj_id(oid, element="Feature", db=self.db)
            test_values = self.transform_test_data(f_doc)
            test_feature = TestFeature(self.obj_id, test_values)
            test_feature.save_db_file(self.db, self.filepaths)

            # record and save
            self.test_features[oid] = test_feature.obj_id
            if update_file:
                ih.save_obj2file(test_feature)
                ih.save_obj2file(self)

            return test_feature.values

    def transform_test_data(self, f_doc):
        lst_fed_oid = f_doc["essentials"]["lst_fed"]
        if lst_fed_oid is None:
            msg = "Are you extrating test features for some initializing features? Those should" + \
                  "be generated in the PipeTestData.__init__, so you might have problem with __init__."
            raise ValueError(msg)

        lst_test_fed_values = []
        for fed_oid in lst_fed_oid:
            lst_test_fed_values.append(self.predict(fed_oid, update_file=False))

        if len(lst_test_fed_values) == 1:
            test_fed_values = lst_test_fed_values[0]
        else:
            test_fed_values = np.concatenate(lst_test_fed_values, axis=1)

        ih = IOHandler()
        mid = f_doc["essentials"]["f_transform"]
        ft = ih.load_obj_from_file(mid, element="FTransform", filepaths=self.filepaths)
        test_values = ft.transform(test_fed_values)

        return test_values

    @staticmethod
    def __get_label(core, test_data):
        lval = None
        col_y = core._y_name
        if bool(col_y) and col_y in test_data.columns:
            lval = test_data[col_y]

        return lval

    @property
    def label(self):
        return self.__label
