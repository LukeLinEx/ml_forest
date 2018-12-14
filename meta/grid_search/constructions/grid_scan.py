import numpy as np
import pandas as pd
from warnings import warn

from ml_forest.core.constructions.db_handler import DbHandler
from ml_forest.core.constructions.io_handler import IOHandler
from ml_forest.pipeline.nodes.stacking_node import FNode, LNode
from ml_forest.pipeline.links.knitor import Knitor

from performance.perf_trackor import PerformanceTrackor
from meta.grid_search.elements.grid_record import GridRecord


class GridScan(object):
    dh = DbHandler()
    ih = IOHandler()

    def __init__(self, grid, evaluator, core_docs, lst_fed, f_transform_type, lnode, target=None, layer=None):
        """

        :param grid: dict or pandas.MultipleIndex or pandas.DataFrame (df of multiple indexes).
                     For a list of dicts, all keys need to be legal to pass to a f_transofrm

        :param evaluator: Evaluator. Metric that evaluates the models
        :param core_docs: ml_forest.core.constructions.core_init.CoreInit
        :param lst_fed: list of FNode
        :param f_transform_type: <type 'type'>
        :param lnode: LNode
        :param target: PipeInit/CoreInit, PipeTestData. If None, use core_docs.
        :param layer
        """
        print("Getting FNodes & LNode ready, can take some time...")
        lst_fed, layer = self.get_features_ready(lst_fed, core_docs, layer)
        self.__layer = layer
        lnode = self.get_label_ready(lnode)

        self.lst_fed = lst_fed
        self.lnode = lnode
        print("Nodes are ready.")

        if not target:
            target = core_docs
        self.target = target

        self.db = core_docs.db
        self.filepaths = core_docs.filepaths

        tmp_grid = GridRecord(evaluator, core_docs, lst_fed, f_transform_type, lnode, target, layer)
        grid_doc = self.search_for_grid_doc(tmp_grid, self.db)
        if grid_doc:
            grid_record = self.ih.load_obj_from_file(grid_doc["_id"], "GridRecord", self.filepaths)
        else:
            grid_record = tmp_grid

        new_grid_idx = self.build_grid_idx(grid)

        # get old grid indexes
        old_grid = grid_record.performance_grid
        if old_grid is not None:
            old_grid_idx = old_grid.index.copy()
        else:
            old_grid_idx = None

        if old_grid_idx is not None:
            expanded_old_grid_idx, expanded_new_grid_idx = self.expand_grid_idxes(
                old_grid_idx, new_grid_idx, grid_record)
            updated_idx = self.append_grid_idxes(expanded_old_grid_idx, expanded_new_grid_idx)

            old_p_grid = grid_record.performance_grid
            expanded_old_grid = self.expand_grid(expanded_old_grid_idx, old_p_grid)
            p_grid = self.update_grid(expanded_old_grid, updated_idx)

            old_r_grid = grid_record.result_grid
            expanded_old_grid = self.expand_grid(expanded_old_grid_idx, old_r_grid)
            r_grid = self.update_grid(expanded_old_grid, updated_idx)
        else:
            p_grid, r_grid = self.create_grid(grid_record, new_grid_idx)

        grid_record.update_performance_grid(p_grid)
        grid_record.update_result_grid(r_grid)
        self.grid_record = grid_record
        self.best_f_id = []

    def update_grid(self, expanded_old_grid, updated_idx):
        expanded_old_grid_idx = expanded_old_grid.index.copy()

        cols = expanded_old_grid.columns.copy()
        new_grid = pd.DataFrame([[np.nan] * len(cols)], index=updated_idx, columns=cols)
        new_grid.loc[expanded_old_grid_idx] = expanded_old_grid.loc[expanded_old_grid_idx]

        return new_grid

    def expand_grid_idxes(self, old_grid_idx, new_grid_idx, grid_record):
        diff_params = set(new_grid_idx.names).symmetric_difference(set(old_grid_idx.names))

        constant_values = {}
        for param in diff_params:
            constant_values[param] = grid_record.f_transform_type().essentials[param]

        old_grid_idx = self.insert_constant_val(old_grid_idx, constant_values)
        new_grid_idx = self.insert_constant_val(new_grid_idx, constant_values)
        new_grid_idx = new_grid_idx.reorder_levels(list(old_grid_idx.names))

        return old_grid_idx, new_grid_idx

    def expand_grid(self, expanded_grid_idx, grid):
        old_params = list(grid.index.names)
        expanded_grid_params = expanded_grid_idx.names

        more_params_included = (set(expanded_grid_params) - set(old_params))
        if not more_params_included:
            result = grid
        else:
            expanded_grid_idx.to_frame().reset_index(drop=True)
            cols = grid.columns.copy()
            tmp_colnames = [str(i) for i in range(grid.shape[1])]

            old_grid = grid.copy()
            old_grid.columns = tmp_colnames
            result = pd.merge(
                old_grid.reset_index(), expanded_grid_idx.to_frame().reset_index(drop=True),
                on=old_params
            ).set_index(expanded_grid_params)

            result.columns = cols

        return result

    def append_grid_idxes(self, old_grid_idx, new_grid_idx):
        old_idx_df = self.idx2idxdf(old_grid_idx)
        new_idx_df = self.idx2idxdf(new_grid_idx)

        grid_idx_df = pd.merge(old_idx_df, new_idx_df, how="outer")
        grid_idx = self.idxdf2idx(grid_idx_df)

        return grid_idx

    def insert_constant_val(self, grid_idx, constant_params):
        idx_df = self.idx2idxdf(grid_idx.copy())
        for param in constant_params:
            if param not in idx_df:
                idx_df[param] = constant_params[param]

        grid_idx = self.idxdf2idx(idx_df)

        return grid_idx

    def create_grid(self, grid_record, idx):
        cols = pd.MultiIndex.from_tuples(
            [(grid_record.evaluator.name, grid_record.target_id)],
            names=["evaluator", "target"]
        )

        p_grid = pd.DataFrame([[np.nan]], index=idx, columns=cols)
        r_grid = pd.DataFrame([[np.nan] * 2], index=idx, columns=["feature_id", "f_transform_id"])

        return p_grid, r_grid

    # testing purpose only
    def save_grid(self):
        self.grid_record.save_db_file(self.db, self.filepaths)

    def update_scheme_obj(self):
        grid_record = self.grid_record
        db = grid_record.core.db
        filepaths = grid_record.core.filepaths

        if grid_record.filepaths:
            self.ih.save_obj2file(grid_record)
        else:
            grid_record.save_db_file(db, filepaths)

    def get_next(self, p_grid):
        remaining_idx = p_grid.loc[p_grid.isnull().sum(axis=1) != 0].index.tolist()

        if remaining_idx:
            return remaining_idx[0]
        else:
            return None

    def grid_scan(self, top_n=1, minimize=True):
        f_transform_type = self.grid_record.f_transform_type
        core = self.grid_record.core
        evaluator = self.grid_record.evaluator
        target = self.target
        db = core.db

        lst_fed = self.lst_fed
        lnode = self.lnode
        # kn = Knitor()

        p_grid = self.grid_record.performance_grid.copy()
        r_grid = self.grid_record.result_grid.copy()
        param_names = p_grid.index.names
        cols_idx = p_grid.columns

        combination = self.get_next(p_grid)

        best_performers = []
        pt = PerformanceTrackor(evaluator, target)

        while bool(combination):
            params = dict(zip(param_names, combination))
            f_transform = f_transform_type(**params)
            f_node = FNode(core, lst_fed, f_transform, lnode)

            _, _, perf, f_id = pt.search_performance(f_node)
            feature = None
            f_transform = None
            if not perf:
                _, _, perf, f_id, feature, f_transform = pt.obtain_performance(f_node)

            doc = self.dh.search_by_obj_id(f_id, "Feature", db)
            ft_id = doc["essentials"]["f_transform"]

            r_grid.loc[combination, "feature_id"] = f_id
            r_grid.loc[combination, "f_transform_id"] = ft_id
            p_grid.loc[combination, cols_idx] = perf

            best_performers.append([feature, f_transform, perf, f_id])

            # TODO: self.save_f&ft_beyond_threshold(f, ft, threshold), this should be an abstract method
            best_performers = sorted(best_performers, key=lambda lst_: (-1)**(minimize+1)*lst_[2])[:top_n]

            combination = self.get_next(p_grid)

        f_2b_saved = [lst[0] for lst in best_performers]
        ft_2b_saved = [lst[1] for lst in best_performers]
        for f in f_2b_saved:
            if f is not None and f.filepaths is None:
                f.save_file(core.filepaths)
        for ft in ft_2b_saved:
            if ft is not None and ft.filepaths is None:
                ft.save_file(core.filepaths)

        self.grid_record.update_performance_grid(p_grid)
        self.grid_record.update_result_grid(r_grid)
        self.update_scheme_obj()

        self.best_f_id = [f_id for _, _, _, f_id in best_performers]

    # Below are trivial
    @staticmethod
    def idx2idxdf(idx):
        idxdf = idx.to_frame().reset_index(drop=True)

        return idxdf

    @staticmethod
    def idxdf2idx(idxdf):
        idx = idxdf.set_index(list(idxdf.columns)).index

        return idx

    def build_grid_idx(self, grid):
        if isinstance(grid, dict):
            idx = pd.MultiIndex.from_product(grid.values(), names=grid.keys())
        elif isinstance(grid, pd.DataFrame):
            idx = self.idxdf2idx(grid)
        elif isinstance(grid, pd.MultiIndex):
            idx = grid
        elif isinstance(grid, pd.RangeIndex):
            idx = grid
        else:
            raise TypeError("Not understanding the object passed to the arguement grid.")

        return idx

    def search_for_grid_doc(self, grid_doc, db):
        """
        Search the GridRecord which recorded old GridScan with the same goal.

        :return: dict or None
        """
        docs = self.dh.search_by_essentials(grid_doc, db)

        if docs:
            print("Old GridRecord found.")
            doc = docs[0]
            if len(docs) > 1:
                warn("We found the same grid search has been done more than once. Check your database.")
            return doc
        else:
            return None

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
