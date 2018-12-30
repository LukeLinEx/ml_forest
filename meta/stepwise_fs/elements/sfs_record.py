from copy import deepcopy
from ml_forest.core.elements.identity import Base
from ml_forest.pipeline.pipe_init import PipeInit


class SFSRecord(Base):
    def __init__(self, evaluator, core_docs, lst_fed, f_transform_type, lnode, target, layer, params):
        lst_fed_id = [fnode.obj_id for fnode in lst_fed]
        if isinstance(target, PipeInit):
            target_id = target.core.obj_id
        else:
            target_id = target.obj_id

        super(SFSRecord, self).__init__()
        self.__f_transform_type = f_transform_type
        self.__essentials = {
            "pipe_init": core_docs.obj_id,
            "evaluator": evaluator.name,
            "lnode": lnode.obj_id,
            "f_transform_type": f_transform_type,
            "target_id": target_id,
            "type": "SFSRecord",  # this prevents the subclasses of Scheme saved differently in db.
            "layer": layer
        }

        for p in params:
            self.__essentials[p] = params[p]

        self.__target_id = target_id
        self.__core = core_docs
        self.__evaluator = evaluator
        self.__result_grid, self.__performance_grid = None, None
        self.__params = params
        self.__layer = layer

        self.__all_candidates = dict(enumerate(lst_fed_id))  # {n: f_id}
        self.__result = []  # {subset, perf, fid, ftid}

    @staticmethod
    def decide_element():
        return "SFSRecord"

    @property
    def all_candidates(self):
        candidates = deepcopy(self.__all_candidates)
        return candidates

    @property
    def result(self):
        result = deepcopy(self.__result)
        return result

    @property
    def params(self):
        params = deepcopy(self.__params)
        return params

    @property
    def f_transform_type(self):
        return self.__f_transform_type

    @property
    def evaluator(self):
        return self.__evaluator

    @property
    def core(self):
        return self.__core

    def add_candidates(self, fed):
        # TODO: allow some other representation of fed features
        n = len(self.all_candidates)
        self.__all_candidates[n] = fed

    def update_result(self, doc):
        self.__result.append(doc)
