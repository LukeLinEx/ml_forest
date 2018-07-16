from ml_forest.core.elements.label_base import Label
from ml_forest.core.elements.feature_base import Feature

from ml_forest.core.constructions.db_handler import DbHandler
from ml_forest.core.constructions.io_handler import IOHandler
from ml_forest.core.constructions.mini_pipe import FFlow, LFlow

from ml_forest.pipeline.nodes.stacking_node import FNode, LNode

# Connectors do not update node's obj_id nor filepaths


class FConnector(object):
    def __init__(self, matched):
        self.matched = matched

    def collect_doc(self, f_node):
        if not isinstance(f_node, FNode):
            raise TypeError("The parameter f_node should of the type FNode.")

        if f_node.obj_id:
            obj_id = f_node.obj_id
            db = f_node.pipe_init.db

            dh = DbHandler()
            doc = dh.search_by_obj_id(obj_id, "Feature", db)
        else:
            lst_f_transform = self.get_f_transform_candidates(f_node)
            all_docs = f_node.get_docs_match_the_fnode(lst_f_transform)
            if all_docs:
                doc = all_docs[0]
            else:
                doc = None

        if doc:
            self.matched.append(doc["essentials"]["f_transform"])

        return doc

    def f_materialize(self, f_node, old_record=None):
        """
        Assuming all the components in this f_node are ready. Generate the Feature and the FTransform based on those.
        The function doesn't search db. Any searched result can be passed to the old_record parameter

        :param f_node:
        :param old_record: a document that matches the node
        :return:
        """
        if not isinstance(f_node, FNode):
            raise TypeError("The parameter f_node should of the type FNode.")

        if old_record:
            if old_record["filepaths"]:
                filepaths = old_record["filepaths"]
                feature_id = old_record["_id"]
                f_transform_id = old_record["essentials"]["f_transform"]

                ih = IOHandler()
                feature = ih.load_obj_from_file(feature_id, "Feature", filepaths)
                f_transform = ih.load_obj_from_file(f_transform_id, "FTransform", filepaths)
            else:
                feature, f_transform = self.recover_with_existing_doc(f_node, old_record)
        else:
            feature, f_transform = self.create_and_record(f_node)

        return feature, f_transform

    # private usage
    def get_f_transform_candidates(self, f_node):
        dh = DbHandler()
        lst_transform_ids = dh.search_by_essentials(f_node.f_transform, f_node.pipe_init.db)
        lst_transform_ids = [x["_id"]for x in lst_transform_ids if x["_id"] not in self.matched]

        return lst_transform_ids

    @staticmethod
    def recover_with_existing_doc(f_node, doc):
        """

        :param f_node:
        :param doc:
        :return:
        """
        db = f_node.pipe_init.db
        filepaths = f_node.pipe_init.filepaths

        frame_id = f_node.pipe_init.frame
        label_id = doc["essentials"]["label"]
        lst_fed_id = [f.obj_id for f in f_node.lst_fed]
        f_transform_id = doc["essentials"]["f_transform"]

        ih = IOHandler()
        frame = ih.load_obj_from_file(frame_id, "Frame", filepaths)
        lst_fed = [ih.load_obj_from_file(f_id, "Feature", filepaths) for f_id in lst_fed_id]
        label = ih.load_obj_from_file(label_id, "Label", filepaths)
        f_transform = f_node.f_transform

        ff = FFlow()
        if f_transform.rise == 1:
            f_values, f_transform, stage = ff.supervised_fit_transform(frame, lst_fed, f_transform, label)
        else:
            f_values, f_transform, stage = ff.unsupervised_fit_transform(frame, lst_fed, f_transform, label)

        f_transform.obj_id = f_transform_id
        f_transform.set_db(db)

        feature = Feature(
            frame=frame_id, f_transform=f_transform_id, label=label_id, lst_fed=lst_fed_id, values=f_values
        )
        feature.stage = stage
        feature.obj_id = doc["_id"]
        feature.set_db(db)

        return feature, f_transform

    @staticmethod
    def create_and_record(f_node):
        """
        Build the object according to the "DNA" in f_node.
        This should be used when no record of the target object is found from the db

        :param f_node:
        :return:
        """
        db = f_node.pipe_init.db
        filepaths = f_node.pipe_init.filepaths

        frame_id = f_node.pipe_init.frame
        label_id = f_node.l_node.obj_id
        lst_fed_id = [f.obj_id for f in f_node.lst_fed]

        ih = IOHandler()
        frame = ih.load_obj_from_file(frame_id, "Frame", filepaths)
        lst_fed = [ih.load_obj_from_file(f_id, "Feature", filepaths) for f_id in lst_fed_id]
        label = ih.load_obj_from_file(label_id, "Label", filepaths)
        f_transform = f_node.f_transform

        ff = FFlow()
        if f_transform.rise == 1:
            f_values, f_transform, stage = ff.supervised_fit_transform(frame, lst_fed, f_transform, label)
        else:
            f_values, f_transform, stage = ff.unsupervised_fit_transform(frame, lst_fed, f_transform, label)

        f_transform.save_db(db)

        feature = Feature(
            frame=frame_id, f_transform=f_transform.obj_id, label=label_id, lst_fed=lst_fed_id, values=f_values
        )
        feature.stage = stage
        feature.save_db(db)

        return feature, f_transform


class LConnector(object):
    def __init__(self, matched):
        self.matched = matched

    def collect_doc(self, l_node):
        if not isinstance(l_node, LNode):
            raise TypeError("The parameter l_node should be of the type LNode.")

        if l_node.obj_id:
            obj_id = l_node.obj_id
            db = l_node.pipe_init.db

            dh = DbHandler()
            doc = dh.search_by_obj_id(obj_id, "Label", db)
        else:
            lst_l_transform = self.get_l_transform_candidates(l_node)
            all_docs = l_node.get_docs_match_the_lnode(lst_l_transform)
            if all_docs:
                doc = all_docs[0]
            else:
                doc = None

        if doc:
            self.matched.append(doc["essentials"]["l_transform"])

        return doc

    def l_materialize(self, l_node, old_record=None):
        """
        Assuming all the components in this l_node are ready. Generate the Label and the LTransform based on those.
        The function doesn't search db. Any searched result can be passed to the old_record parameter

        :param l_node: LNode
        :param old_record: a document matched the node.
        :return:
        """
        if not isinstance(l_node, LNode):
            raise TypeError("The parameter l_node should be of the type LNode.")

        if old_record:
            if old_record["filepaths"]:
                filepaths = old_record["filepaths"]
                label_id = old_record["_id"]
                l_transform_id = old_record["essentials"]["l_transform"]

                ih = IOHandler()
                label = ih.load_obj_from_file(label_id, "Label", filepaths)
                l_transform = ih.load_obj_from_file(l_transform_id, "LTransform", filepaths)
            else:
                label, l_transform = self.recover_with_existing_doc(l_node, old_record)
        else:
            label, l_transform = self.create_and_record(l_node)

        return label, l_transform

    # private usage
    def get_l_transform_candidates(self, l_node):
        dh = DbHandler()
        lst_transform_ids = dh.search_by_essentials(l_node.l_transform, l_node.pipe_init.db)
        lst_transform_ids = [x["_id"]for x in lst_transform_ids if x["_id"] not in self.matched]
        return lst_transform_ids

    @staticmethod
    def recover_with_existing_doc(l_node, doc):
        """
        From the document we found, recover the Label and the LTransform object.
        This should be used when a record is found in the db but the object itself is not saved

        :param doc:
        :param l_node:
        :return:
        """
        db = l_node.pipe_init.db
        filepaths = l_node.pipe_init.filepaths

        frame_id = l_node.pipe_init.frame
        lab_fed_id = l_node.lab_fed.obj_id
        l_transform_id = doc["essentials"]["l_transform"]

        ih = IOHandler()
        frame = ih.load_obj_from_file(frame_id, "Frame", filepaths)
        lab_fed = ih.load_obj_from_file(lab_fed_id, "Label", filepaths)
        l_transform = l_node.l_transform

        lflow = LFlow()
        l_values, l_transform = lflow.label_encoding_transform(frame, lab_fed, l_transform)

        l_transform.obj_id = l_transform_id
        l_transform.set_db(db)

        label = Label(frame=frame_id, l_transform=l_transform_id, raw_y=lab_fed_id, values=l_values)
        label.set_db(db)
        label.obj_id = doc["_id"]

        return label, l_transform

    @staticmethod
    def create_and_record(l_node):
        """
        Build the object according to the "DNA" in l_node.
        This should be used when no record of the target object is found from the db

        :param l_node:
        :return:
        """
        db = l_node.pipe_init.db
        filepaths = l_node.pipe_init.filepaths

        frame_id = l_node.pipe_init.frame
        lab_fed_id = l_node.lab_fed.obj_id

        ih = IOHandler()
        frame = ih.load_obj_from_file(frame_id, "Frame", filepaths)
        lab_fed = ih.load_obj_from_file(lab_fed_id, "Label", filepaths)
        l_transform = l_node.l_transform

        lflow = LFlow()
        l_values, l_transform = lflow.label_encoding_transform(frame, lab_fed, l_transform)

        l_transform.save_db(db)
        label = Label(frame=frame_id, l_transform=l_transform.obj_id, raw_y=lab_fed_id, values=l_values)
        label.save_db(db)

        return label, l_transform
