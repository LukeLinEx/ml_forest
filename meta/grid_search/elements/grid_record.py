from ml_forest.core.elements.identity import Base
from ml_forest.core.constructions.io_handler import IOHandler

from ml_forest.pipeline.pipe_init import PipeInit
from ml_forest.pipeline.nodes.stacking_node import FNode


class GridRecord(Base):
    def __init__(self, evaluator, core_docs, lst_fed, f_transform_type, lnode, target, layer):
        """

        :param evaluator: Evaluator. Metric that evaluates the models
        :param core_docs: ml_forest.core.constructions.core_init.CoreInit
        :param lst_fed: list of FNode
        :param f_transform_type: <type 'type'>
        :param lnode: LNode
        :param target: PipeInit/CoreInit, PipeTestData.
        :param
        """
        lst_fed_id = [fnode.obj_id for fnode in lst_fed]
        if isinstance(target, PipeInit):
            target_id = target.core.obj_id
        else:
            target_id = target.obj_id

        super(GridRecord, self).__init__()
        self.__f_transform_type = f_transform_type
        self.__essentials = {
            "pipe_init": core_docs.obj_id,
            "evaluator": evaluator.name,
            "lst_fed": lst_fed_id,
            "lnode": lnode.obj_id,
            "f_transform_type": f_transform_type,
            "target_id": target_id,
            "type": "GridRecord",  # this prevents the subclasses of Scheme saved differently in db.
            "layer": layer
        }

        self.__target_id = target_id
        self.__core = core_docs
        self.__evaluator = evaluator
        self.__result_grid, self.__performance_grid = None, None
        self.__lst_params = []
        self.__layer = layer

    @staticmethod
    def decide_element():
        return "GridRecord"

    @property
    def core(self):
        return self.__core

    @property
    def label(self):
        filepaths = self.__core.filepaths
        lid = self.__essentials["lnode"]

        ih = IOHandler()
        return ih.load_obj_from_file(lid, "Label", filepaths)

    @property
    def layer(self):
        return self.__layer

    @property
    def lst_fed(self):
        filepaths = self.__core.filepaths

        lst_fed = self.__essentials["lst_fed"]
        lst_fed = [FNode(core=self.__core, obj_id=fid, filepaths=filepaths) for fid in lst_fed]

        return lst_fed

    @property
    def frame(self):
        filepaths = self.__core.filepaths
        _id = self.__core.frame

        ih = IOHandler()
        return ih.load_obj_from_file(_id, "Frame", filepaths)

    @property
    def evaluator(self):
        return self.__evaluator

    @property
    def f_transform_type(self):
        return self.__f_transform_type

    @property
    def target_id(self):
        return self.__target_id

    @property
    def lst_params(self):
        return self.__lst_params[:]

    @property
    def performance_grid(self):
        return self.__performance_grid

    def update_performance_grid(self, new_grid):
        self.__performance_grid = new_grid

    @property
    def result_grid(self):
        return self.__result_grid

    def update_result_grid(self, new_grid):
        self.__result_grid = new_grid
