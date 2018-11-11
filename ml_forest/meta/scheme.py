from bson.objectid import ObjectId
import pandas as pd

from ml_forest.core.elements.identity import Base
from ml_forest.core.constructions.db_handler import DbHandler
from ml_forest.core.constructions.io_handler import IOHandler

from ml_forest.pipeline.nodes.stacking_node import FNode, LNode
from ml_forest.pipeline.links.knitor import Knitor


# TODO: this is a temperary version. We need a thorough refactor
# TODO: When a extended Scheme is scanned, we need to remove the old best performers who are no long in our best list
class Scheme(Base):
    def __init__(self, layer, grid_dict, evaluators, core_docs, lst_fed, f_transform_type, lnode):
        """
        !!!!!!! EXTENDIBLILITY !!!!!!!!!!
        !!!!!!!!!! The output should be a series of (feature, f_transform)s


        :param layer: int, TODO: allow derive from lst_fed in the future
        :param grid_dict: list of dicts. All keys need to be legal to pass to a f_transofrm
        :param evaluators: lst of evaluating functions
        :param core_docs: ml_forest.core.constructions.core_init.CoreInit
        :param lst_fed: list of obj_id
        :param lnode: obj_id
        :param f_transform_type: <type 'type'>
        """
        for fed in lst_fed:
            if not isinstance(fed, FNode) or (fed.obj_id is None):
                raise TypeError("Currently can only accept FNode with obj_id (fitted) as features")

        if not isinstance(lnode, LNode) or lnode.obj_id is None:
            raise TypeError("Currently can only accept LNode with obj_id.")

        super(Scheme, self).__init__()
        self.__f_transform_type = f_transform_type
        self.__essentials = {
            "pipe_init": core_docs.obj_id,
            # "evaluators": evaluators,
            "lst_fed": [f.obj_id for f in lst_fed],
            "lnode": lnode.obj_id,
            "f_transform_type": f_transform_type,
            "type": "Scheme"  # this prevents the subclasses of Scheme saved differently in db.
        }

        self.__core = core_docs

        self.__layer = layer
        self.__evaluators = evaluators
        self.__result_grid, self.__performance_grid = self.create_grid(grid_dict)

    def create_grid(self, grid_dict):
        frame_id = self.__core.frame
        filepaths = self.__core.filepaths

        ih = IOHandler()
        frame = ih.load_obj_from_file(frame_id, "Frame", filepaths)
        if frame.depth < self.__layer:
            raise ValueError("The input layer can't be more than the depth of the frame.")

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

        if docs:
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

        new_performance_grid = pd.concat([old_performance_grid, new_performance_grid])
        new_result_grid = pd.concat([old_result_grid, new_result_grid])

        self.update_grid(
            pgrid=new_performance_grid,
            rgrid=new_result_grid
        )

    #     def insert_new_index(self, grid_df, grid_dict):
    #         old_grid = grid_df.copy()
    #         old_idx = old_grid.index.to_frame()

    #         for key in grid_dict:
    #             if key not in old_idx.columns:
    #                 old_idx[key] = self.return_constant_params(key)

    #         val = old_grid.values
    #         idx = pd.MultiIndex.from_arrays(old_idx.values.T, names=old_idx.columns)
    #         updated = pd.DataFrame(val, index=idx, columns=old_grid.columns)

    #         return updated

    def insert_new_index(self, grid_df, grid_dict):
        old_grid = grid_df.copy()
        old_idx = old_grid.index.to_frame()

        for key in grid_dict:
            if key not in old_idx.columns:
                old_idx[key] = self.return_constant_params(key)

        val = old_grid.values
        names = old_idx.columns.tolist()
        old_idx = old_idx.set_index(old_idx.columns.tolist(), drop=True)
        idx = pd.MultiIndex.from_tuples(old_idx.index.tolist(), names=names)
        updated = pd.DataFrame(val, index=idx, columns=old_grid.columns)

        return updated

    def return_constant_params(self, key):
        """

        At this point, self should be loaded from a old record/storage
        For the keys that are not in grid_dict, find the values from ftransform.essentials[key]

        :param key:
        :return:
        """
        param = self.f_transform_type().essentials[key]

        dh = DbHandler()

        checklst = []
        for ft in self.result_grid["f_transform_id"]:
            if isinstance(ft, ObjectId):
                doc = dh.search_by_obj_id(ft, "FTransform", self.db)
                checklst.append(doc["essentials"][key])
        if [p for p in checklst if p != param]:
            raise TypeError("Something very wrong in this package.")

        return param

    @staticmethod
    def none_repeat_index(old, new):
        old_ = old.copy()
        old_["repeat"] = True

        new_ = new.copy()
        new_["non_repeat"] = old_["repeat"]

        non_repeat_idx = new_["non_repeat"].isnull()
        return non_repeat_idx

    def update_grid(self, pgrid=None, rgrid=None):
        if pgrid is not None:
            self.__performance_grid = pgrid

        if rgrid is not None:
            self.__result_grid = rgrid

    # Filling the trained result
    def update_scheme_obj(self):
        ih = IOHandler()
        ih.save_obj2file(self)

    def get_starter(self):
        raise NotImplementedError

    def get_next(self):
        raise NotImplementedError

    def grid_scan(self, top_n=1, main_eval=None):
        if self.obj_id is None:
            raise ValueError(
                "It seems you create a general Scheme object. Use a derived class that specifies a particular process.")

        names = self.performance_grid.index.names
        label = self.label
        lst_fed = self.lst_fed
        frame = self.frame
        core = self.__core

        fold_idx = frame.create_structure(self.layer)
        combination = self.get_starter()

        if main_eval is None:
            main_eval = self.evaluators[0].__name__

        best_performers = []
        while bool(combination):
            params = dict(zip(names, combination))
            f_transform = self.__f_transform_type(**params)
            lnode = LNode(core=self.__core, obj_id=self.essentials["lnode"], filepaths=self.__core.filepaths)
            f_node = FNode(core, lst_fed, f_transform, lnode)
            kn = Knitor()
            feature, f_transform = kn.f_knit(f_node)

            self.__result_grid.loc[combination, "feature_id"] = feature.obj_id
            self.__result_grid.loc[combination, "f_transform_id"] = feature.essentials["f_transform"]

            for idx in fold_idx:
                rows = frame.get_single_fold(idx)
                f = feature.values[rows, :]
                l = label.values[rows, :]

                for evaluator in self.evaluators:
                    perf = evaluator(f, l)
                    self.__performance_grid[(evaluator.__name__, idx)].loc[combination] = perf

            avg_perf = self.__performance_grid[main_eval].loc[combination].mean()
            best_performers.append([feature, f_transform, avg_perf])
            best_performers = sorted(best_performers, key=lambda lst_: lst_[2])[:top_n]

            # TODO: self.save_f&ft_beyond_threshold(f, ft, threshold), this should be an abstract method
            combination = self.get_next()

        f_2b_saved = [lst[0] for lst in best_performers]
        ft_2b_saved = [lst[1] for lst in best_performers]
        for f in f_2b_saved:
            if f.filepaths is None:
                f.save_file(self.__core.filepaths)
        for ft in ft_2b_saved:
            if ft.filepaths is None:
                ft.save_file(self.__core.filepaths)

        self.update_scheme_obj()

        return best_performers

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
    def evaluators(self):
        return self.__evaluators

    @property
    def f_transform_type(self):
        return self.__f_transform_type


class SimpleGridSearch(Scheme):
    def __init__(self, layer, grid_dict, evaluators, core_docs, lst_fed, f_transform_type, lnode):
        super(SimpleGridSearch, self).__init__(
            layer, grid_dict, evaluators, core_docs, lst_fed, f_transform_type, lnode
        )
        obj = Scheme(layer, grid_dict, evaluators, core_docs, lst_fed, f_transform_type, lnode)
        obj = obj.search_for_scheme(core_docs.db)

        if bool(obj):
            print("An old Scheme object is found and used")
            self.__dict__.update(obj.__dict__)
            self.expand_grids(grid_dict)
            try:
                self.__essentials
            except AttributeError:
                self.__essentials = {}
                self.update_scheme_obj()
        else:
            self.__essentials = {}
            self.save_db_file(core_docs.db, core_docs.filepaths)

    def get_starter(self):
        # change to list to prevent parameters casted to numpy.int64 which causes problems search in mongo
        starter = list(self.performance_grid.index)[0]
        return starter

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

