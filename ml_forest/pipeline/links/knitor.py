"""
Knitors update node's obj_id and filepaths, but Knitors should update node.filepaths only if the obj is found saved in
the filepaths already. Whether or not to save a newly obtain object should be decided outside of the Knitors.
"""

from ml_forest.core.constructions.io_handler import IOHandler
from ml_forest.pipeline.links.connector import FConnector, LConnector


# TODO: in f_(sub)knit, returning f_transform makes no sense since fnode has that already
class Knitor(object):
    """

    Each Knitor instance should be used for one full training task (to be defined more precisely, an typical training
    task could be a grid search). A new knitor re-initialize `matched`, which prevents from looking into those matched
    with other features already. Reuse a Knitor instance will miss existing features in searching.
    """
    def __init__(self):
        matched = {
            "f": [], "l": []
        }
        self.fc = FConnector(matched["f"])
        self.lc = LConnector(matched["l"])

    def l_subknit(self, l_node):
        # TODO: need to prevent subnitting fitted l_transform somehow
        if not l_node.filepaths:
            if l_node.lab_fed:
                self.l_subknit(l_node.lab_fed)

            doc = self.lc.collect_doc(l_node)
            if doc and "filepaths" in doc and doc["filepaths"]:
                obj_id = doc["_id"]
                filepaths = doc["filepaths"]
            else:
                filepaths = l_node.core.filepaths

                label, l_transform = self.lc.l_materialize(l_node, doc)

                label.save_file(filepaths)
                l_transform.save_file(filepaths)

                obj_id = label.obj_id

            l_node.filepaths = filepaths
            if l_node.obj_id is None:
                l_node.obj_id = obj_id

    def l_knit(self, l_node):
        # TODO: need to prevent knitting fitted l_transform somehow
        if l_node.lab_fed:
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
            if f_node.lst_fed:
                for f in f_node.lst_fed:
                    self.f_subknit(f)
            if f_node.l_node:
                self.l_subknit(f_node.l_node)

            doc = self.fc.collect_doc(f_node)
            if doc and "filepaths" in doc and doc["filepaths"]:
                filepaths = doc["filepaths"]
                obj_id = doc["_id"]
            else:
                # TODO: Current f_materialize doesn't work with non-empty ftransform. Fix this and rm the next line
                self.fnode_has_empty_ftransform(f_node)

                filepaths = f_node.core.filepaths

                feature, f_transform = self.fc.f_materialize(f_node, doc)
                feature.save_file(filepaths)
                f_transform.save_file(filepaths)

                obj_id = feature.obj_id

            f_node.filepaths = filepaths
            if f_node.obj_id is None:
                f_node.obj_id = obj_id

    def f_knit(self, f_node):
        if f_node.lst_fed:
            for f in f_node.lst_fed:
                self.f_subknit(f)
        if f_node.l_node:
            self.l_subknit(f_node.l_node)

        if f_node.filepaths:
            ih = IOHandler()
            feature = ih.load_obj_from_file(f_node.obj_id, "Feature", f_node.filepaths)
            f_transform_id = feature.essentials["f_transform"]
            f_transform = ih.load_obj_from_file(f_transform_id, "FTransform", f_node.filepaths)
        else:
            # TODO: Current f_materialize doesn't work with non-empty ftransform. Fix this and rm the next line
            self.fnode_has_empty_ftransform(f_node)

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

    @staticmethod
    def fnode_has_empty_ftransform(f_node):
        """
        A Knitor object update a f_transform (the models attribute, in particular)or a in the node through
        the knitting process. This function inspect if the f_transform can be updated (in particular, check
        if the models attribute is still empty).

        :param f_node: stacking_nodes.FNode
        :return:
        """
        if f_node.f_transform and f_node.f_transform.models is not None:
            msg = "Currently Knitor can only knit an empty f_transform. The f_transform in the f_node you passed " +\
                  "has been fit already. There should be a pair of Feature and FTransform objects you got when you " +\
                  "last knitted this node, which usually give you everything you need. Or you can create a new " +\
                  "node with an empty f_transform."
            raise ValueError(msg)
