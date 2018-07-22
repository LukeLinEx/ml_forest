import os
from bson.objectid import ObjectId
from ml_forest.pipeline.pipe_init import PipeInit

home = os.path.expanduser("~")
bucket = "mltests3mongo"
home_path = home + "/Desktop/test_ml_forest/experiment/local_storage"
project = "housing_price"

db = {"host":bucket, "project":project}
filepaths = [{"home": home_path, "project":project}]


# TODO: the test actually depends on PipeInit, should take it into account as well.
pipe_id = ObjectId("5b54ef67b6492933d34f7ed8")
pipe_init = PipeInit(pipe_id=pipe_id, filepaths=filepaths)
core_docs = pipe_init.core
init_fnodes = pipe_init.init_fnodes
init_lnode = pipe_init.init_lnode


import unittest

key = 'LandContour'

from ml_forest.core.utils.connect_mongo import connect_collection

from ml_forest.pipeline.nodes.stacking_node import FNode, LNode
from ml_forest.pipeline.links.connector import FConnector
from ml_forest.pipeline.links.knitor import Knitor

from ml_forest.feature_transformations.encoding.simple_dummy import SimpleDummy


class testKnittingNodesWithDummy(unittest.TestCase):
    def clean_feature(self):
        a_node = FNode(core_docs, [init_fnodes[key]], SimpleDummy(), init_lnode)
        fc = FConnector(matched=[])
        a_doc = fc.collect_doc(a_node)

        if a_doc:
            f_id = a_doc["_id"]
            ft_id = a_doc["essentials"]["f_transform"]

            f_collection = connect_collection(bucket, project, "Feature")
            f_collection.delete_one({"_id": f_id})
            ft_collection = connect_collection(bucket, project, "FTransform")
            ft_collection.delete_one({"_id": ft_id})

            filepaths = a_doc["filepaths"]
            if filepaths:
                for path in filepaths:
                    if "home" in path:
                        _path = [path[k] for k in path]
                        fpath = "/".join(_path + ["Feature", str(f_id) + ".pkl"])
                        ftpath = "/".join(_path + ["FTransform", str(ft_id) + ".pkl"])
                        os.remove(fpath)
                        os.remove(ftpath)
                    elif "bucket" in path:
                        _path = [path[k] for k in path]
                        fpath = "/".join(_path + ["Feature", str(f_id) + ".pkl"])
                        ftpath = "/".join(_path + ["FTransform", str(ft_id) + ".pkl"])
                        raise NotImplementedError
                    else:
                        raise ValueError("Path unknown")
        return a_node

    def KnittingWOdocNORfilepaths(self):
        self.clean_feature()
        a_node = FNode(core_docs, [init_fnodes[key]], SimpleDummy(), init_lnode)
        kn = Knitor()
        kn.f_knit(a_node)
        fc = FConnector(matched=[])
        a_doc = fc.collect_doc(a_node)
        self.assertTrue("_id" in a_doc)
        self.assertTrue("f_transform" in a_doc["essentials"])
        self.assertEqual(len(a_doc["filepaths"]), 0)

    def KnittingWdocNORfilepaths(self):
        a_node = FNode(core_docs, [init_fnodes[key]], SimpleDummy(), init_lnode)
        kn = Knitor()
        kn.f_knit(a_node)
        fc = FConnector(matched=[])
        a_doc = fc.collect_doc(a_node)
        self.assertTrue("_id" in a_doc)
        self.assertTrue("f_transform" in a_doc["essentials"])
        self.assertEqual(len(a_doc["filepaths"]), 0)

    def SubKnittingWdocNORfilepaths(self):
        a_node = FNode(core_docs, [init_fnodes[key]], SimpleDummy(), init_lnode)
        kn = Knitor()
        kn.f_subknit(a_node)
        fc = FConnector(matched=[])
        a_doc = fc.collect_doc(a_node)
        self.assertTrue("_id" in a_doc)
        self.assertTrue("f_transform" in a_doc["essentials"])
        self.assertNotEqual(len(a_doc["filepaths"]), 0)

    def KnittingWdocANDfilepaths(self):
        a_node = FNode(core_docs, [init_fnodes[key]], SimpleDummy(), init_lnode)
        kn = Knitor()
        kn.f_knit(a_node)
        fc = FConnector(matched=[])
        a_doc = fc.collect_doc(a_node)

    def SubKnittingWdocANDfilepaths(self):
        a_node = FNode(core_docs, [init_fnodes[key]], SimpleDummy(), init_lnode)
        kn = Knitor()
        kn.f_knit(a_node)
        fc = FConnector(matched=[])
        a_doc = fc.collect_doc(a_node)

    def SubKnittingWOdocNORfilepaths(self):
        self.clean_feature()
        a_node = FNode(core_docs, [init_fnodes[key]], SimpleDummy(), init_lnode)
        kn = Knitor()
        kn.f_subknit(a_node)
        fc = FConnector(matched=[])
        a_doc = fc.collect_doc(a_node)
        self.assertTrue("_id" in a_doc)
        self.assertTrue("f_transform" in a_doc["essentials"])
        self.assertNotEqual(len(a_doc["filepaths"]), 0)

    def testone(self):
        self.KnittingWOdocNORfilepaths()
        self.KnittingWdocNORfilepaths()
        self.SubKnittingWdocNORfilepaths()
        self.KnittingWdocANDfilepaths()
        self.SubKnittingWdocANDfilepaths()
        self.SubKnittingWOdocNORfilepaths()


if __name__ ==  "__main__":
    unittest.main(argv = ["first-arg-is-ignored"], exit=False)
