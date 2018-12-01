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

    def predict(self, feature, save_prediction=False, update_pipe=False):
        """
        Return the prediction/transformation corresponding to the feature.

        :param feature: FNode or Feature. Feature has to be with a fitted FTransform.
        :param save_prediction: boolean, if the prediction, TestFeature, is to be saved
        :param update_pipe: boolean, if the PipeTestData is to be updated
        :return:
        """
        if not isinstance(feature, Feature) and not isinstance(feature, FNode):
            msg = "the feature has to be of the Feature or the FNode class."
            raise TypeError(msg)
        if isinstance(feature, Feature) and not feature.filepaths:
            msg = "If a Feature is passed to the feature argument, " + \
                  "the Feature object needs to saved (have filepaths attribute) already."
            raise TypeError(msg)

        # If the test feature saved already
        if feature.obj_id and self.test_feature_exists(feature.obj_id):
            oid = feature.obj_id
            _id = self.pipe.test_features[oid]
            test_feature = self.ih.load_obj_from_file(_id, "TestFeature", self.pipe.filepaths)
        # If the test feature not saved yet
        else:
            f, f_transform, test_values = self.transform_test_data(feature)
            test_feature = TestFeature(self.pipe.obj_id, test_values)

            # Decide how to save the prediction (TestFeature)
            if save_prediction:
                if not f.filepaths:
                    f.save_file(self.pipe.filepaths)
                if not f_transform.filepaths:
                    f_transform.save_file(self.pipe.filepaths)
                if isinstance(feature, Feature) and not feature.filepaths:
                    feature.save_file(self.pipe.filepaths)
                if isinstance(feature, FNode) and not feature.filepaths:
                    feature.filepaths = self.pipe.filepaths

                test_feature.save_db_file(self.pipe.db, self.pipe.filepaths)
                self.pipe.test_features[feature.obj_id] = test_feature.obj_id

        # Update the PipeTestData object in the files
        if update_pipe:
            self.ih.save_obj2file(self.pipe)

        return test_feature

    def collect_f_transform(self, feature):
        pass

    def transform_test_data(self, feature):
        # Collect f_transform and lst_fed for either FNode or Feature
        if isinstance(feature, FNode):
            if not feature.lst_fed:
                msg = "Are you extrating test features for some initializing features? Those should" + \
                      "be generated in the PipeTestData.__init__, so you might have problem with __init__."
                raise ValueError(msg)

            kn = Knitor()
            feature, f_transform = kn.f_knit(feature)  # After this line, feature become of Feature Type (not FNode)
            lst_fed_oid = feature.essentials["lst_fed"]
        else:
            if not feature.essentials["lst_fed"]:
                msg = "Are you extrating test features for some initializing features? Those should" + \
                      "be generated in the PipeTestData.__init__, so you might have problem with __init__."
                raise ValueError(msg)

            ft_id = feature.essentials["f_transform"]
            f_transform = self.ih.load_obj_from_file(ft_id, "FTransform", feature.filepaths)
            lst_fed_oid = feature.essentials["lst_fed"]

        if not lst_fed_oid:
            msg = "Are you extrating test features for some initializing features? Those should" + \
                  "be generated in the PipeTestData.__init__, so you might have problem with __init__."
            raise ValueError(msg)

        lst_test_fed_values = []
        lst_fed = [self.ih.load_obj_from_file(fid, "Feature", self.pipe.filepaths) for fid in lst_fed_oid]
        for fed in lst_fed:
            fed_val = self.predict(fed, save_prediction=True, update_pipe=False).values
            lst_test_fed_values.append(fed_val)

        if len(lst_test_fed_values) == 1:
            test_fed_values = lst_test_fed_values[0]
        else:
            test_fed_values = np.concatenate(lst_test_fed_values, axis=1)

        test_values = f_transform.transform(test_fed_values)

        return feature, f_transform, test_values
