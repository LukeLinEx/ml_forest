import numpy as np
import pandas as pd
from bson.objectid import ObjectId

from ml_forest.core.elements.identity import Base

class Scheme(Base):
    def __init__(self, layer, grid_dict, evaluators, pipe_init, frame, lst_fed, method_type, label, **kwargs):
        """
        !!!!!!! EXTENDIBLILITY !!!!!!!!!!
        !!!!!!!!!! The output should be a series of (feature, method)s


        :param layer: int, TODO: allow derive from lst_fed in the future
        :param grid: list of dicts. All keys need to be legal to pass to a method
        :param evaluators: lst of stacking_station.pipe_line.evaluate_funcs.Evaluators
        :param frame: big_fundamental.frame_base.Frame
        :param lst_fed: list of big_fundamental.feature_base.Feature.obj_id
                        All in the lst_fed are assumed trained.
                        TODO: One day lift this so FNode can be passed.
        :param label: big_fundamental.label_base.Label or
                      big_fundamental.pipe_line.stacking_nodes.LNode
                      label is assumed transformed properly already, so obj_id should be ready

        :param method_type: <type 'type'> of some big_fundamental.method_base.Method.
                       Notice that this is callable
        :param kwargs: saving related. Check the attribute in SaveBase
        """
        for fed in lst_fed + [label]:
            if not isinstance(fed, ObjectId):
                raise TypeError("currently can only accept ObjectId of a label or a feature")
        assert frame.depth >= layer, "The input layer can't be more than the depth of the frame"

        super(Scheme, self).__init__(**kwargs)
        self.__frame = frame
        self.__method_type = method_type
        self.__essentials = {
            "pipe_init": pipe_init.obj_id,
            "frame": frame.obj_id,
            "lst_fed": lst_fed,
            "label": label,
            "method_type": method_type,
            "type": "Scheme"
        }

        self.__layer = layer
        self.__evaluators = evaluators
        self.__result_grid, self.__performance_grid = self.create_grid(grid_dict)