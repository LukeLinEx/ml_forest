import numpy as np

from ml_forest.core.constructions.io_handler import IOHandler
from ml_forest.core.constructions.db_handler import DbHandler

from ml_forest.core.elements.feature_base import Feature
from ml_forest.pipeline.nodes.stacking_node import FNode
from ml_forest.pipeline.links.knitor import Knitor

from test_data.elements.pipe_test_data import PipeTestData
from test_data.elements.test_feature import TestFeature


class Piper(object):
    def __init__(self, pipe_test_data):
        if not isinstance(pipe_test_data, PipeTestData):
            raise TypeError("pipe_test_data has to be of the PipeTestData class")

        self.pipe = pipe_test_data
        self.ih = IOHandler()
        self.dh = DbHandler()

    def test_feature_exists(self, oid):
        return oid in self.pipe.test_features

    def predict_with_fnode(self, fnode):
        """
        Return the prediction/transformation corresponding to the fnode.

        :param fnode: FNode
        :return:
        """
        # If the test feature saved already
        if fnode.obj_id and self.test_feature_exists(fnode.obj_id):
            print("Get the test feature from storage.")
            oid = fnode.obj_id
            _id = self.pipe.test_features[oid]
            test_feature = self.ih.load_obj_from_file(_id, "TestFeature", self.pipe.filepaths)
            feature = None
            f_transform = None
            fid = fnode.obj_id
        # If the test feature not saved yet
        else:
            print("Build the test feature")
            if not fnode.lst_fed:
                msg = "Are you extrating test features for some initializing features? Those should" + \
                      "be generated in the PipeTestData.__init__, so you might have problem with __init__."
                raise ValueError(msg)

            kn = Knitor()
            feature, f_transform = kn.f_knit(fnode)  # After this line, feature become of Feature Type (not FNode)
            fid = feature.obj_id

            if self.test_feature_exists(fid):
                _id = self.pipe.test_features[fid]
                test_feature = self.ih.load_obj_from_file(_id, "TestFeature", self.pipe.filepaths)
            else:
                lst_fed_oid = feature.essentials["lst_fed"]
                test_fed_values = self.get_test_fed_values(lst_fed_oid)
                test_values = f_transform.transform(test_fed_values)
                test_feature = TestFeature(self.pipe.obj_id, test_values)

        return test_feature, feature, f_transform, fid

    def predict_with_feature(self, feature):
        """
        Return the prediction/transformation corresponding to the feature.

        :param feature: Feature with a fitted FTransform.
        :return:
        """
        if isinstance(feature, Feature) and not feature.filepaths:
            msg = "If a Feature is passed to the feature argument, " + \
                  "the Feature object needs to saved (have filepaths attribute) already."
            raise TypeError(msg)

        # If the test feature saved already
        if self.test_feature_exists(feature.obj_id):
            print("Get the test feature from storage.")
            oid = feature.obj_id
            _id = self.pipe.test_features[oid]
            test_feature = self.ih.load_obj_from_file(_id, "TestFeature", self.pipe.filepaths)
            f_transform = None
        # If the test feature not saved yet
        else:
            print("Build the test feature")
            lst_fed_oid = feature.essentials["lst_fed"]
            if not lst_fed_oid:
                msg = "Are you extrating test features for some initializing features? Those should" + \
                      "be generated in the PipeTestData.__init__, so you might have problem with __init__."
                raise ValueError(msg)
            test_fed_values = self.get_test_fed_values(lst_fed_oid)

            ft_id = feature.essentials["f_transform"]
            f_transform = self.ih.load_obj_from_file(ft_id, "FTransform", feature.filepaths)

            test_values = f_transform.transform(test_fed_values)
            test_feature = TestFeature(self.pipe.obj_id, test_values)

        fid = feature.obj_id

        return test_feature, feature, f_transform, fid

    def get_test_fed_values(self, lst_fed_oid):
        lst_test_fed_values = []
        lst_fed = [self.ih.load_obj_from_file(fid, "Feature", self.pipe.filepaths) for fid in lst_fed_oid]
        for fed in lst_fed:
            fed_val = self.predict(fed, save_prediction=True, update_pipe=False).values
            lst_test_fed_values.append(fed_val)

        if len(lst_test_fed_values) == 1:
            test_fed_values = lst_test_fed_values[0]
        else:
            test_fed_values = np.concatenate(lst_test_fed_values, axis=1)

        return test_fed_values

    def predict(self, f_, save_prediction=False, update_pipe=False):
        """
        Return the prediction/transformation corresponding to the feature.
        Note: if an FNode passed to f_, it's used as a reference to find TestFeatue, no gaurantee if
              knit/subknit happened. Might need to reproduce the KNode for knitting the Feature

        :param f_: FNode or Feature. Feature has to be with a fitted FTransform.
        :param save_prediction: boolean, if the prediction, TestFeature, is to be saved
        :param update_pipe: boolean, if the PipeTestData is to be updated
        :return:
        """
        if isinstance(f_, Feature):
            test_feature, feature, f_transform, fid = self.predict_with_feature(f_)
        elif isinstance(f_, FNode):
            test_feature, feature, f_transform, fid = self.predict_with_fnode(f_)
        else:
            msg = "the feature has to be of the Feature or the FNode class."
            raise TypeError(msg)

        if save_prediction:
            db = self.pipe.db
            filepaths = self.pipe.filepaths
            if feature and not feature.filepaths:
                feature.save_file(filepaths)
            if f_transform and not f_transform.filepaths:
                f_transform.save_file(filepaths)
            if not self.test_feature_exists(fid):
                test_feature.save_db_file(db, filepaths)
                self.pipe.test_features[feature.obj_id] = test_feature.obj_id

        # Update the PipeTestData object in the files
        if update_pipe:
            self.ih.save_obj2file(self.pipe)

        return test_feature
