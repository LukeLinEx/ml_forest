from bson.objectid import ObjectId
from ml_forest.core.constructions.core_init import CoreInit
from ml_forest.core.constructions.db_handler import DbHandler
from ml_forest.core.constructions.io_handler import IOHandler
from ml_forest.pipeline.nodes.stacking_node import FNode
from ml_forest.pipeline.links.knitor import Knitor
from performance.evaluators import Evaluator
from test_data.constructions.piper import Piper


class PerformanceTrackor(object):
    def __init__(self, evaluator, target, label_id=None):
        """

        :param evaluator: Evaluator
        :param target: PipeInit/CoreInit or PipeTestData
        :param label_id: ObjectId that represents the Label object. If None, the label in target would be used
        """
        if not isinstance(evaluator, Evaluator):
            raise TypeError("Only accept performance.evaluators.Evaluator as the evaluator.")

        self.ev_name = evaluator.name
        self.ev_func = evaluator

        if not hasattr(target, "label"):
            raise AttributeError("The target doesn't have a label attribute for evaluation.")
        if isinstance(target.label, ObjectId):
            self.target_type = "self"
            if isinstance(target, CoreInit):
                self.target_id = target.obj_id
            else:
                self.target_id = target.core.obj_id
        else:
            self.target_type = "test"
            self.target_id = target.obj_id

        self.target = target

        self.label_id = None
        if label_id:
            self.label_id = label_id

    def search_performance_by_id(self, fid, db=None):
        """

        :param fid: ObjectId
        :param db: dict, indicates the database to search from.
                   Required if a Feature or an ObjectId is passed to f
        """
        dh = DbHandler()
        doc = dh.search_by_obj_id(fid, "Feature", db)

        performance = None
        if doc and "performance" in doc:
            performance_lst = doc["performance"]
            performance_docs = [
                subdoc for subdoc in performance_lst if subdoc["target"] == self.target_id and
                                                        subdoc["ev_name"] == self.ev_name and
                                                        subdoc["label"] == self.label_id
                ]
            if performance_docs:
                performance = performance_docs[0]["score"]

        return performance

    def get_performance(self, fval):
        ih = IOHandler()
        if self.label_id:
            label = ih.load_obj_from_file(self.label_id, "Label", self.target.filepaths)
            lval = label.values
        elif self.target_type == "self":
            label = ih.load_obj_from_file(self.target.label, "Label", self.target.filepaths)
            lval = label.values
        else:
            lval = self.target.label

        performance = self.ev_func(fval, lval)
        return performance

    def get_label_id(self):
        if self.label_id:
            return self.label_id
        else:
            label_obj = self.target.label
            if isinstance(label_obj, ObjectId):
                return self.label_id
            else:
                return None

    def record_performance(self, f_id, perf):
        db = self.target.db
        label_id = self.get_label_id()
        subdoc = {"target": self.target_id, "label": label_id, "ev_name": self.ev_name, "score": perf}

        dh = DbHandler()
        dh.insert_subdoc_by_id(f_id, "Feature", db, "performance", subdoc)

    def search_performance(self, f, core=None):
        """

        :param f: Feature, FNode or ObjectId, from where obj_id to search is obtained
        :param core:
        :return:
        """
        if not core and not isinstance(f, FNode):
            raise TypeError("If core is not provided, f has to be of FNode type")
        elif isinstance(f, FNode):
            db = f.core.db
        else:
            db = core.db

        if isinstance(f, ObjectId):
            f_id = f
        elif isinstance(f, FNode):
            kn = Knitor()
            doc = kn.fc.collect_doc(f)
            if doc is not None:
                f_id = doc["_id"]
            else:
                f_id = None
        else:
            try:
                f_id = f.obj_id
                if not f_id:
                    raise AttributeError("f has empty obj_id, can't search the Feature.")
            except AttributeError:
                raise AttributeError("f doesn't have the obj_id attribute for searching the Feature object.")

        perf = self.search_performance_by_id(f_id, db)

        return self.ev_name, self.target_id, perf, f_id

    def obtain_performance(self, f):
        if not isinstance(f, FNode):
            msg = "To obtain the Feature for evaluation, f has to be a FNode"
            raise ValueError(msg)

        if self.target_type == "test":
            pred, feature, f_transform = Piper(self.target).predict(f)
            f_id = f.obj_id
        else:
            kn = Knitor()
            pred, f_transform = kn.f_knit(f)
            f_id = pred.obj_id
            feature = pred

        fval = pred.values
        perf = self.get_performance(fval)
        self.record_performance(f.obj_id, perf)

        return self.ev_name, self.target_id, perf, f_id, feature, f_transform
