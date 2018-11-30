from bson.objectid import ObjectId
from ml_forest.pipeline.nodes.stacking_node import FNode
from ml_forest.core.constructions.db_handler import DbHandler
from ml_forest.core.constructions.io_handler import IOHandler
from ml_forest.pipeline.links.knitor import Knitor
from performance import evaluators

ev_module_name = evaluators.__name__


class PerformanceTrackor(object):
    def __init__(self, evaluator, target):
        """

        :param evaluator: an evaluating function from performance.evaluators
        :param target: PipeInit or PipeTestData
        """
        if evaluator.__module__ != ev_module_name:
            raise TypeError("Only accept evaluators from {}".format(ev_module_name))

        self.ev_name = evaluator.__name__
        self.evaluate_func = evaluator

        if not hasattr(target, "label"):
            raise AttributeError("The target doesn't have a label attribute for evaluation.")
        if isinstance(target.label, ObjectId):
            self.target_type = "self"
            self.target_id = target.core.obj_id
        else:
            self.target_type = "test"
            self.target_id = target.obj_id

        self.target = target

    def search_performance(self, f, db=None):
        """

        :param f: Feature, FNode or ObjectId, from where obj_id to search is obtained
        :param db: dict, indicates the database to search from.
                   Required if a Feature or an ObjectId is passed to f
        """
        dh = DbHandler()
        doc = dh.search_by_obj_id(f.obj_id, "Feature", db)

        performance = None
        if "performance" in doc:
            performance_lst = doc["performance"]
            performance_docs = [
                subdoc for subdoc in performance_lst
                if subdoc["target"] == self.target_id and subdoc["ev_name"] == self.ev_name
                ]
            if performance_docs:
                performance = performance_docs[0]["score"]

        return performance

    def get_performance(self, fnode):
        if self.target_type == "self":
            ih = IOHandler()
            label = ih.load_obj_from_file(self.target.label, "Label", self.target.filepaths)
            lval = label.values
        else:
            lval = self.target.label

        kn = Knitor()
        f, ft = kn.f_knit(fnode)
        fval = f.values

        performance = self.evaluate_func(fval, lval)
        return performance

    def record_performance(self, f_id, perf):
        """
        """
        db = self.target.db
        subdoc = {"target": self.target_id, "ev_name": self.ev_name, "score": perf}

        dh = DbHandler()
        dh.insert_subdoc_by_id(f_id, "Feature", db, "performance", subdoc)

    def output_performance(self, f, core=None):
        if not core and not isinstance(f, FNode):
            raise TypeError("If core is not provided, f has to be of FNode type")
        elif isinstance(f, FNode):
            db = f.core.db
        else:
            db = core.db

        perf = self.search_performance(f, db)

        if not perf and not isinstance(f, FNode):
            msg = "The performance was not found from the db. To create the Feature, " + \
                  "f has to be a FNode"
            raise ValueError(msg)
        elif not perf:
            print(perf)
            perf = self.get_performance(f)
            self.record_performance(f.obj_id, perf)

        return perf
