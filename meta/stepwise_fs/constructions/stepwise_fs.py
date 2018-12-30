from copy import deepcopy
import numpy as np
from warnings import warn
from ml_forest.core.constructions.db_handler import DbHandler
from ml_forest.core.constructions.io_handler import IOHandler

from ml_forest.pipeline.nodes.stacking_node import FNode, LNode
from ml_forest.pipeline.links.knitor import Knitor

from performance.perf_trackor import PerformanceTrackor
from meta.stepwise_fs.elements.sfs_record import SFSRecord


# TODO: if Feature and FTransform were not split into two classes, do I have problem here?
class StepwiseFeatureSelection(object):
    def __init__(
            self, evaluator, core_docs, lst_fed, f_transform_type, lnode, target=None, layer=None,
            omit_old=False, keep_untouched=None, **kwargs
    ):
        self.dh = DbHandler()
        self.ih = IOHandler()
        self.db = core_docs.db
        self.filepaths = core_docs.filepaths

        # TODO: should check if keep_untouched is in the sfs_record.result already. Warn if not.
        if not keep_untouched:
            keep_untouched = set()
        self.keep_untouched = keep_untouched

        # check if kwargs are params for f_transform
        try:
            f_transform_type(**kwargs)
            params = kwargs
        except TypeError as e:
            bad_arg = str(e).split("__init__() got an unexpected keyword argument ")[1]
            msg = "The f_transform_type got an unexpected keyword argument {}".format(bad_arg)
            raise TypeError(msg)

        # TODO: this should go beyond and shared by all meta
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

        # find old sfs if any, then update the candidate
        if omit_old:
            sfs_record = SFSRecord(evaluator, core_docs, lst_fed, f_transform_type, lnode, target, layer, params)
        else:
            sfs_record = self.obtain_sfs_record(
                evaluator, core_docs, lst_fed, f_transform_type, lnode, target, layer, params
            )

        old_candidates = set(sfs_record.all_candidates.values())
        for fed in lst_fed:
            if fed.obj_id not in old_candidates:
                sfs_record.add_candidates(fed.obj_id)

        self.sfs_record = sfs_record
        self.best_performers = []

    def stepwise_search(
            self, num_try_b4_stop, num_test_each_step, top_n=1, minimize=True, update_increment=30
    ):
        all_candidates = self.sfs_record.all_candidates

        sign = (-1) ** (minimize + 1)
        ev_name = self.sfs_record.evaluator.name
        tmp = sorted(self.sfs_record.result, key=lambda doc_: sign * doc_[ev_name])[:top_n]
        best_performers = deepcopy(tmp)

        for doc in best_performers:
            doc["feature"] = None
            doc["f_transform"] = None
        lst_included = self.get_lst_included(best_performers)

        i = 0
        j = 0
        while i < num_try_b4_stop:
            testing_features = np.random.choice(range(len(all_candidates)), num_test_each_step, replace=False)
            testing_features = set(testing_features)

            best_performers = self.searching_within_a_step(
                all_candidates, testing_features, best_performers, top_n, minimize)

            new_lst_included = self.get_lst_included(best_performers)
            if lst_included == new_lst_included:
                i += 1
            else:
                lst_included = new_lst_included
                i = 0
            j += 1
            print(j)
            if j % update_increment == 0:
                self.update_sfs_record(save_best=False)
                print("another update...")

        self.best_performers = best_performers
        print("last update")
        self.update_sfs_record(best_performers)

    def get_lst_included(self, best_performers):
        if best_performers:
            lst_included = [doc["included"] for doc in best_performers]
        else:
            lst_included = []

        return lst_included

    def searching_within_a_step(self, num2fnodes, testing_features, best_performers, top_n, minimize):
        db = self.db
        evaluator = self.sfs_record.evaluator
        ev_name = evaluator.name
        core = self.sfs_record.core
        params = self.sfs_record.params
        f_transform_type = self.sfs_record.f_transform_type
        target = self.target
        lnode = self.lnode

        lst_included = self.get_lst_included(best_performers)
        if lst_included:
            included = lst_included[0]
        else:
            included = set()

        pt = PerformanceTrackor(evaluator, target)

        for n in testing_features:
            if n in self.keep_untouched:
                continue

            if n in included:
                testing = included - {n}
            else:
                testing = included.union({n})

            if not testing:
                continue

            been_tested = [doc["included"] for doc in self.sfs_record.result]
            if testing in been_tested:
                continue

            testing_fed_id = [num2fnodes[n] for n in sorted(list(testing))]
            testing_fed = [FNode(core, obj_id=f_id, filepaths=core.filepaths) for f_id in testing_fed_id]
            f_transform = f_transform_type(**params)

            # TODO: vvv go to upper level and shared by all meta
            f_node = FNode(core, testing_fed, f_transform, lnode)

            _, _, perf, f_id = pt.search_performance(f_node)
            feature = None
            f_transform = None
            if not perf:
                _, _, perf, f_id, feature, f_transform = pt.obtain_performance(f_node)

            doc = self.dh.search_by_obj_id(f_id, "Feature", db)
            ft_id = doc["essentials"]["f_transform"]
            # TODO: ^^^ go to upper level and shared by all meta

            result_doc = {"included": testing, ev_name: perf, "feature_id": f_id, "f_transform_id": ft_id}
            self.sfs_record.update_result(result_doc)

            result_doc["feature"] = feature
            result_doc["f_transform"] = f_transform
            sign = (-1) ** (minimize + 1)
            best_performers.append(result_doc)
            best_performers = sorted(best_performers, key=lambda doc_: sign * doc_[ev_name])[:top_n]

        return best_performers

    def obtain_sfs_record(self, evaluator, core_docs, lst_fed, f_transform_type, lnode, target, layer, params):
        tmp_sfs = SFSRecord(evaluator, core_docs, lst_fed, f_transform_type, lnode, target, layer, params)
        sfs_record = self.search_for_sfs_record(tmp_sfs, self.db)
        if sfs_record:
            sfs_record = self.ih.load_obj_from_file(sfs_record["_id"], "SFSRecord", self.filepaths)
        else:
            sfs_record = tmp_sfs

        return sfs_record

    def search_for_sfs_record(self, sfs_record, db):
        """
        Search the SFSRecord which recorded old StepwiseFeatureSelection with the same goal.

        :return: dict or None
        """
        docs = self.dh.search_by_essentials(sfs_record, db)

        if docs:
            print("Old SFSRecord found.")
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

    def update_sfs_record(self, best_performers=None, save_best=True):
        core = self.sfs_record.core
        sfs_record = self.sfs_record
        db = sfs_record.core.db
        filepaths = sfs_record.core.filepaths

        if sfs_record.filepaths:
            self.ih.save_obj2file(sfs_record)
        else:
            sfs_record.save_db_file(db, filepaths)

        if save_best:
            f_2b_saved = [doc["feature"] for doc in best_performers]
            ft_2b_saved = [doc["f_transform"] for doc in best_performers]
            for f in f_2b_saved:
                if f is not None and not f.filepaths:
                    f.save_file(core.filepaths)
            for ft in ft_2b_saved:
                if ft is not None and not ft.filepaths:
                    ft.save_file(core.filepaths)
