from bson.objectid import ObjectId

from ml_forest.core.constructions.io_handler import IOHandler
from ml_forest.core.constructions.core_init import CoreInit

from ml_forest.pipeline.nodes.stacking_node import FNode, LNode

# TODO: need a much better way to inspect input

class PipeInit(object):
    def __init__(self, data=None, col_y=None, lst_layers=None, shuffle=False, stratified=False, col_selected=None,
                 tag=None, db=None, filepaths=None, pipe_id=None):
        """
        :param data: pandas.DataFrame. This needs to be a pandas data frame with a label column
        :param col_y: The name of the label column
        :param lst_layers: list. This gives the "lst_layers" to the Frame
        :param shuffle: boolean.
        :param stratified: boolean. Should not be used to a regression problem
        :param col_selected: dict. Ex: {'num': ['colname1', 'colname2'], 'cate':['colname3'], ...}
        :param db:
        :param filepaths:
        :param pipe_id
        """
        if pipe_id and isinstance(pipe_id, ObjectId) and filepaths:
            ih = IOHandler()
            self.core = ih.load_obj_from_file(obj_id=pipe_id, element="CoreInit", filepaths=filepaths)
        elif pipe_id and not isinstance(pipe_id, ObjectId):
            raise TypeError("The pipe_id you passed is not an ObjectId.")
        else:
            self.core = CoreInit(data, col_y, lst_layers, shuffle, stratified, col_selected, tag, db, filepaths)

        init_fnodes = self.init_features
        for key in init_fnodes:
            init_fnodes[key] = FNode(self.core, obj_id=init_fnodes[key])
        self._init_fnodes = init_fnodes

        init_lnode = LNode(self.core, obj_id=self.label)
        self._init_lnode = init_lnode

    @property
    def init_features(self):
        return self.core.init_features

    @property
    def frame(self):
        return self.core.frame

    @property
    def label(self):
        return self.core.label

    @property
    def db(self):
        return self.core.db

    @property
    def filepaths(self):
        return self.core.filepaths

    @property
    def init_fnodes(self):
        return self._init_fnodes

    @property
    def init_lnode(self):
        return self._init_lnode


if __name__ == "__main__":
    pass
