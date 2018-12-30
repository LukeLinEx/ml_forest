from copy import deepcopy
import numpy as np
from warnings import warn

from ml_forest.pipeline.nodes.stacking_node import FNode

from performance.perf_trackor import PerformanceTrackor
from meta.stepwise_fs.elements.sfs_record import SFSRecord
from meta.meta import Meta


# TODO: if Feature and FTransform were not split into two classes, do I have problem here?
class StepwiseFeatureSelection(Meta):
    def __init__(
            self, evaluator, core_docs, lst_fed, f_transform_type, lnode, target=None, layer=None,
            omit_old=False, keep_untouched=None, **kwargs
    ):
        super(StepwiseFeatureSelection, self).__init__(core_docs)

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

        self.lst_fed, self.lnode, layer = self.get_nodes_ready(lst_fed, core_docs, layer, lnode)
        self.__layer = layer
        target = self.get_target_ready(target, core_docs)
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

    def get_best_performers(self, lst_docs, sign, ev_name, top_n):
        obp = sorted(lst_docs, key=lambda doc_: sign * doc_[ev_name])[:top_n]
        obp = deepcopy(obp)

        return obp

    def get_lst_included(self, best_performers):
        if best_performers:
            lst_included = [doc["included"] for doc in best_performers]
        else:
            lst_included = []

        return lst_included

    def stepwise_search(
            self, num_try_b4_stop, num_test_each_step, top_n=1, minimize=True, update_increment=30
    ):
        all_candidates = self.sfs_record.all_candidates

        sign = self.get_sign(minimize)
        ev_name = self.sfs_record.evaluator.name
        best_performers = self.get_best_performers(self.sfs_record.result, sign, ev_name, top_n)

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
                all_candidates, testing_features, best_performers, top_n, sign)

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

    def searching_within_a_step(self, num2fnodes, testing_features, best_performers, top_n, sign):
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

            f_node = FNode(core, testing_fed, f_transform, lnode)
            perf, f_id, ft_id, feature, f_transform = self.evaluate(f_node, pt)

            # updating relevant records
            result_doc = {"included": testing, ev_name: perf, "feature_id": f_id, "f_transform_id": ft_id}
            self.sfs_record.update_result(result_doc)
            result_doc["feature"] = feature
            result_doc["f_transform"] = f_transform

            best_performers.append(result_doc)
            best_performers = self.get_best_performers(best_performers, sign, ev_name, top_n)

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
            self.save_best_performers2files(best_performers, core)

    def save_best_performers2files(self, best_performers, core):
        if not best_performers:
            msg = "The NoneType best_performers can't be saved."
            raise TypeError(msg)

        f_2b_saved = [doc["feature"] for doc in best_performers]
        ft_2b_saved = [doc["f_transform"] for doc in best_performers]
        for f in f_2b_saved:
            if f is not None and not f.filepaths:
                f.save_file(core.filepaths)
        for ft in ft_2b_saved:
            if ft is not None and not ft.filepaths:
                ft.save_file(core.filepaths)
