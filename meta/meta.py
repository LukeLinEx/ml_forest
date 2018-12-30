from ml_forest.core.constructions.db_handler import DbHandler
from ml_forest.core.constructions.io_handler import IOHandler

from ml_forest.pipeline.nodes.stacking_node import FNode, LNode
from ml_forest.pipeline.links.knitor import Knitor


class Meta(object):
    def __init__(self, core_docs):
        self.dh = DbHandler()
        self.ih = IOHandler()
        self.db = core_docs.db
        self.filepaths = core_docs.filepaths

    def get_nodes_ready(self, lst_fed, core_docs, layer, lnode):
        print("Getting FNodes & LNode ready, can take some time...")
        lst_fed, layer = self.get_features_ready(lst_fed, core_docs, layer)
        lnode = self.get_label_ready(lnode)
        print("Nodes are ready.")

        return lst_fed, lnode, layer

    @staticmethod
    def get_features_ready(lst_fed, core_docs, layer):
        filepaths = core_docs.filepaths

        kn = Knitor()
        for fed in lst_fed:
            if not isinstance(fed, FNode):
                raise TypeError("Only takes FNode as an input variable.")
            if not fed.obj_id or not fed.filepaths:
                kn.f_subknit(fed)

        lst_fed_id = [f.obj_id for f in lst_fed]

        if not layer:
            ih = IOHandler()
            current_stage = max(ih.load_obj_from_file(fid, "Feature", filepaths).stage for fid in lst_fed_id)
            depth = ih.load_obj_from_file(core_docs.frame, "Frame", filepaths).depth
            layer = depth - current_stage

            if layer < 0:
                raise ValueError(
                    "Negative value is not possible. Check if there is mismatch between lst_fed and core_docs")

        return lst_fed, layer

    @staticmethod
    def get_label_ready(lnode):
        if not isinstance(lnode, LNode):
            raise TypeError("Only takes LNode as an output variable.")
        if not lnode.obj_id or not lnode.filepaths:
            kn = Knitor()
            kn.l_subknit(lnode)

        return lnode

    def get_target_ready(self, target, core_docs):
        if not target:
            target = core_docs

        return target

    def get_best_performers(self, **kwargs):
        raise NotImplementedError

    def evaluate(self, f_node, pt):
        db = f_node.core.db
        _, _, perf, f_id = pt.search_performance(f_node)
        feature = None
        f_transform = None
        if not perf:
            _, _, perf, f_id, feature, f_transform = pt.obtain_performance(f_node)

        doc = self.dh.search_by_obj_id(f_id, "Feature", db)
        ft_id = doc["essentials"]["f_transform"]

        return perf, f_id, ft_id, feature, f_transform

    @staticmethod
    def get_sign(minimize):
        sign = (-1) ** (minimize + 1)

        return sign
