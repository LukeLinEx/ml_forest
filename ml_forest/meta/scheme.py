import numpy as np
import pandas as pd
from bson.objectid import ObjectId

from ml_forest.core.elements.identity import Base
from ml_forest.core.constructions.db_handler import DbHandler
from ml_forest.core.constructions.io_handler import IOHandler

from ml_forest.pipeline.stacking_node import FNode, LNode
from ml_forest.pipeline.nodes_pack import Connector


# TODO: this is a temperary version. We need a thorough refactor
class Scheme(Base):
    def __init__(self, layer, grid_dict, evaluators, pipe_init, frame, lst_fed, f_transform_type, label):
        """
        !!!!!!! EXTENDIBLILITY !!!!!!!!!!
        !!!!!!!!!! The output should be a series of (feature, f_transform)s


        :param layer: int, TODO: allow derive from lst_fed in the future
        :param grid_dict: list of dicts. All keys need to be legal to pass to a f_transofrm
        :param evaluators: lst of evaluating functions
        :param pipe_init: obj_id
        :param frame: obj_id
        :param lst_fed: list of obj_id
        :param label: obj_id
        :param f_transform_type: <type 'type'>
        """
        for fed in lst_fed + [label]:
            if not isinstance(fed, ObjectId):
                raise TypeError("currently can only accept ObjectId of a label or a feature")
        assert frame.depth >= layer, "The input layer can't be more than the depth of the frame"

        super(Scheme, self).__init__()
        self.__f_transform_type = f_transform_type
        self.__essentials = {
            "pipe_init": pipe_init.obj_id,
            "frame": frame,
            "lst_fed": lst_fed,
            "label": label,
            "f_transform_type": f_transform_type
        }

        # TODO: need to reconsider this. Currently this is the only obj (not obj_id) saved in Scheme
        self.__pipe_init = pipe_init

        self.__layer = layer
        self.__evaluators = evaluators
        self.__result_grid, self.__performance_grid = self.create_grid(grid_dict)

    def create_grid(self, grid_dict):
        frame_id = self.__essentials["frame"]
        filepaths = self.__pipe_init.filepaths
        ih = IOHandler()
        frame = ih.load_obj_from_file(frame_id, "Frame", filepaths)

        idx = pd.MultiIndex.from_product(grid_dict.values(), names=grid_dict.keys())
        folds = frame.create_structure(self.__layer)
        evals = [e.__name__ for e in self.__evaluators]
        cols = pd.MultiIndex.from_product([evals, folds])

        r_grid, p_grid = pd.DataFrame(index=idx, columns=["feature_id", "f_transform_id"]), \
                         pd.DataFrame(index=idx, columns=cols)

        return r_grid, p_grid

    def search_for_scheme(self, db):
        """

        :return:
        """
        dh = DbHandler()
        docs = dh.search_by_essentials(self, db)

        if bool(docs):
            doc = docs[0]
            obj_id = doc["_id"]
            filepaths = doc["filepaths"]
            element = self.decide_element()

            ih = IOHandler()
            scheme_loaded = ih.load_obj_from_file(obj_id, element, filepaths)
            return scheme_loaded
        else:
            return None

    @property
    def performance_grid(self):
        return self.__performance_grid

    @property
    def result_grid(self):
        return self.__result_grid

    @staticmethod
    def decide_element():
        return "Scheme"

    def expand_grids(self, grid_dict):
        old_performance_grid = self.insert_new_index(self.performance_grid, grid_dict)
        old_result_grid = self.insert_new_index(self.result_grid, grid_dict)

        m = self.essentials["f_transform_type"]()
        all_grid_keys = old_performance_grid.index.names
        for key in all_grid_keys:
            if key not in grid_dict:
                grid_dict[key] = [m.essentials[key]]

        new_result_grid, new_performance_grid = self.create_grid(grid_dict)
        idx = self.none_repeat_index(old_result_grid, new_result_grid)

        new_performance_grid = new_performance_grid.loc[idx]
        new_result_grid = new_result_grid.loc[idx]

        new_performance_grid = pd.concat(
            [old_performance_grid.reset_index(), new_performance_grid.reset_index()]
        ).set_index(all_grid_keys)

        new_result_grid = pd.concat(
            [old_result_grid.reset_index(), new_result_grid.reset_index()]
        ).set_index(all_grid_keys)

        self.update_grid(
            pgrid=new_performance_grid,
            rgrid=new_result_grid
        )

    def insert_new_index(self, grid_df, grid_dict):
        old_grid = grid_df.copy()
        old_keys = list(old_grid.index.names)
        old_grid = old_grid.reset_index()

        added = []
        for key in grid_dict:
            if key not in old_grid.columns:
                added.append(key)
                old_grid[key] = self.return_constant_params(key)

        return old_grid.set_index(old_keys + added)

    def return_constant_params(self, key):
        """

        At this point, self should be loaded from a old record/storage
        For the keys that are not in grid_dict, find the values from self.essentials[key]

        :param key:
        :return:
        """
        filepaths = self.pipe_init.filepaths

        param_lst = []
        ih = IOHandler()
        for ft_id in self.result_grid["f_transform_id"]:
            f_transform = ih.load_obj_from_file(ft_id, "FTransform", filepaths)
            param_lst.append(f_transform.essentials[key])

        if len(set(param_lst)) > 1:
            raise ValueError("Something seriously wrong with the design of the Scheme family")
        else:
            return param_lst[0]

    @staticmethod
    def none_repeat_index(old, new):
        idx = list(old.index.names)
        old = old.index.to_frame().reset_index(drop=True)
        new = new.index.to_frame().reset_index(drop=True)

        old["repeat"] = True

        repeat = pd.merge(new, old, how="left", on=idx)
        repeat = repeat.replace(np.nan, False).set_index(idx)

        return np.logical_not(repeat["repeat"])

    def update_grid(self, pgrid=None, rgrid=None):
        if pgrid is not None:
            self.__performance_grid = pgrid

        if rgrid is not None:
            self.__result_grid = rgrid

    # Filling the trained result
    def update_scheme(self):
        ih = IOHandler()
        ih.save_obj2file(self)

    def get_starter(self):
        raise NotImplementedError

    def get_next(self):
        raise NotImplementedError

    def grid_scan(self):
        if self.obj_id is None:
            raise ValueError(
                "It seems you create a general Scheme object. Use a derived class that specifies a particular process.")
        names = self.performance_grid.index.names
        label = self.label
        lst_fed = self.lst_fed
        frame = self.frame
        pipe_init = self.__pipe_init

        fold_idx = frame.create_structure(self.layer)
        combination = self.get_starter()

        while bool(combination):
            params = dict(zip(names, combination))
            f_transform = self.__f_transform_type(**params)
            fnode = FNode(pipe_init, lst_fed, f_transform, label)
            connector = Connector()
            connector.locate(fnode)


            fnode.search()
            feature = Feature.get_obj_from_cls_filepaths(self.filepaths, fnode.obj_id)
            label = Label.get_obj_from_cls_filepaths(self.filepaths, feature.label)
            self.__result_grid.loc[combination, "feature_id"] = feature.obj_id
            self.__result_grid.loc[combination, "f_transform_id"] = feature.f_transform

            for idx in fold_idx:
                rows = frame.get_single_fold(idx)
                f = feature.values[rows, :]
                l = label.values[rows]

                for evaluator in self.evaluators:
                    self.__performance_grid[(evaluator.name, idx)].loc[combination] = evaluator.evaluate(f, l)

            self.update_scheme()
            combination = self.get_next()

    @property
    def label(self):
        filepaths = self.__pipe_init.filepaths
        lid = self.__essentials["label"]

        ih = IOHandler()
        return ih.load_obj_from_file(lid, "Label", filepaths)

    @property
    def layer(self):
        return self.__layer

    @property
    def lst_fed(self):
        filepaths = self.__pipe_init.filepaths
        lst_fed = self.__essentials["lst_fed"]

        ih = IOHandler()
        return [ih.load_obj_from_file(fid, "Feature", filepaths) for fid in lst_fed]

    @property
    def frame(self):
        filepaths = self.__pipe_init.filepaths
        _id = self.__essentials["frame"]

        ih = IOHandler()
        return ih.load_obj_from_file(_id, "Frame", filepaths)

    @property
    def evaluators(self):
        return self.__evaluators


class SimpleGridSearch(Scheme):
    def __init__(self, layer, grid_dict, evaluators, pipe_init, frame, lst_fed, f_transform_type, label):
        super(SimpleGridSearch, self).__init__(
            layer, grid_dict, evaluators, pipe_init, frame, lst_fed, f_transform_type, label
        )
        obj = Scheme(layer, grid_dict, evaluators, pipe_init, frame, lst_fed, f_transform_type, label)
        obj = obj.search_for_scheme(pipe_init.db)

        if bool(obj):
            print("An old Scheme object is found and used")
            self.__dict__.update(obj.__dict__)
            self.expand_grids(grid_dict)
            try:
                self.__essentials
            except AttributeError:
                self.__essentials = {}
                self.update_scheme()
        else:
            self.__essentials = {}
            self.save_db_file(pipe_init.db, pipe_init.filepaths)

    def get_starter(self):
        return self.performance_grid.index[0]

    def get_next(self):
        remain = list(
            self.performance_grid.loc[
                self.performance_grid.isnull().sum(axis=1) != 0
            ].index
        )
        if remain:
            return remain[0]
        else:
            return None


# if __name__ == "__main__":
#     from sklearn.metrics import accuracy_score, log_loss
#     from ml_forest.core.elements.frame_base import Frame
#
#     evaluators = [accuracy_score, log_loss]
#     frame = Frame(203, [2, 3, 5])
#     grid_dict = {"a": [1,2,3], "b":["x", "y", "z"]}
#     s = Scheme(0, grid_dict, evaluators, frame, frame, [], [ObjectId()], ObjectId())
#
#     print(
#         s.performance_grid
#     )
#
