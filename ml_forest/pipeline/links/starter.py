from copy import deepcopy

from ml_forest.pipeline.nodes.stacking_node import FNode, LNode
from ml_forest.pipeline.pipe_init import PipeInit


class Starter(object):
    @staticmethod
    def get_init_nodes(pinit):
        if not isinstance(pinit, PipeInit):
            raise TypeError("The starter only works with pipeline.pipe_init.PipeInit.")

        init_fnodes = deepcopy(pinit.init_features)
        for key in init_fnodes:
            init_fnodes[key] = FNode(pinit, obj_id=init_fnodes[key])

        init_lnode = LNode(pinit, obj_id=pinit.label)

        return init_fnodes, init_lnode
