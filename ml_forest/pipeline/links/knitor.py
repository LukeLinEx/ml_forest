"""
Knitors update node's obj_id and filepaths, but Knitors should update node.filepaths only if the obj is found saved in
the filepaths already. Whether or not to save a newly obtain object should be decided outside of the Knitors.
"""

from ml_forest.core.constructions.io_handler import IOHandler
from ml_forest.pipeline.links.connector import FConnector, LConnector


class Knitor(object):
    def __init__(self):
        matched = {
            "f": [], "l": []
        }
        self.fc = FConnector(matched["f"])
        self.lc = LConnector(matched["l"])

    def l_subknit(self, l_node):
        if not l_node.filepaths:
            self.l_subknit(l_node.lab_fed)

            doc = self.lc.collect_doc(l_node)
            if doc and "filepaths" in doc:
                obj_id = doc["_id"]
                filepaths = doc["filepaths"]
            else:
                filepaths = l_node.pipe_init.filepaths

                label, l_transform = self.lc.l_materialize(l_node, doc)

                label.save_file(filepaths)
                l_transform.save_file(filepaths)

                obj_id = label.obj_id

            l_node.filepaths = filepaths
            if l_node.obj_id is None:
                l_node.obj_id = obj_id

    def l_knit(self, l_node):
        self.l_subknit(l_node.lab_fed)

        if l_node.filepaths:
            ih = IOHandler()
            label = ih.load_obj_from_file(l_node.obj_id, "Label", l_node.filepaths)
            l_transform_id = label.l_transform
            l_transform = ih.load_obj_from_file(l_transform_id, "LTransform", l_node.filepaths)
        else:
            doc = self.lc.collect_doc(l_node)
            label, l_transform = self.lc.l_materialize(l_node, doc)
            if doc and "filepaths" in doc:
                """"
                Update if the obj is already saved in the filepaths.
                Whether or not save a new created one should be decided by a higher level function
                """
                l_node.filepaths = doc["filepaths"]

                obj_id = doc["_id"]
            else:
                obj_id = label.obj_id

            if l_node.obj_id is None:
                l_node.obj_id = obj_id

        return label, l_transform

    def f_subknit(self, f_node):
        if not f_node.filepaths:
            for f in f_node.lst_fed:
                self.f_subknit(f)

            self.l_subknit(f_node.l_node)

            doc = self.fc.collect_doc(f_node)
            if doc and "filepaths" in doc:
                filepaths = doc["filepaths"]
                obj_id = doc["_id"]
            else:
                filepaths = f_node.pipe_init.filepaths

                feature, f_transform = self.fc.f_materialize(f_node, doc)
                feature.save_file(filepaths)
                f_transform.save_file(filepaths)

                obj_id = feature.obj_id

            f_node.filepaths = filepaths
            if f_node.obj_id is None:
                f_node.obj_id = obj_id

    def f_knit(self, f_node):
        for f in f_node.lst_fed:
            self.f_subknit(f)

        self.l_subknit(f_node.l_node)

        if f_node.filepaths:
            ih = IOHandler()
            feature = ih.load_obj_from_file(f_node.obj_id, "Feature", f_node.filepaths)
            f_transform_id = feature.f_transform
            f_transform = ih.load_obj_from_file(f_transform_id, "FTransform", f_node.filepaths)
        else:
            doc = self.fc.collect_doc(f_node)
            feature, f_transform = self.fc.f_materialize(f_node, doc)
            if doc and "filepaths" in doc:
                """
                Update if the obj is already saved in the filepaths.
                Whether or not save a new created one should be decided by a higher level function
                """
                f_node.filepaths = doc["filepaths"]

                obj_id = doc["_id"]
            else:
                obj_id = feature.obj_id

            if f_node.obj_id is None:
                f_node.obj_id = obj_id

        return feature, f_transform
