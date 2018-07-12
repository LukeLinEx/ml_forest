from ml_forest.core.elements.feature_base import Feature
from ml_forest.core.elements.label_base import Label
from ml_forest.core.constructions.db_handler import DbHandler
from ml_forest.core.constructions.mini_pipe import MiniPipe

from ml_forest.pipeline.stacking_node import FNode, LNode

# TODO: The Assembler should be used with only PipeInit. Be careful!!
# class Assembler(object):
#     def __init__(self):
#         self.nodes = []
#
#     def collect_and_sort(self, obj_id_dict):
#         obj_id_prlst = sorted(obj_id_dict.items(), key=lambda p: p[0])
#         obj_id_lst = [p[1] for p in obj_id_prlst]
#         return obj_id_lst
#
#     def get_nodes(self, pipe_init, obj_id_dict):
#         obj_id_lst = self.collect_and_sort(obj_id_dict)
#         nodes = [FNode(pipe_init=pipe_init, obj_id=oid) for oid in obj_id_lst]
#         self.nodes = nodes


class Connector(object):
    def __init__(self):
        self.matched = {
            "f": [], "l": []
        }

    def locate(self, node, save_obtained=True):
        if isinstance(node, FNode):
            fc = FConnector(self.matched)
            fc.f_locate(node, save_obtained)
        elif isinstance(node, LNode):
            lc = LConnector(self.matched["t"])
            lc.l_locate(node, save_obtained)


class LConnector(object):
    def __init__(self, matched):
        self.matched = matched

    def l_locate(self, l_node, save_obtained=True):
        if not isinstance(l_node, LNode):
            raise TypeError("The parameter l_node should be of the type LNode.")

        label_obtained, l_trans_obtained = None, None
        db = l_node.pipe_init.db
        filepaths = l_node.pipe_init.filepaths

        if l_node.obj_id is None:
            lst_l_transform = self.l_prepare_locate(l_node)
            all_docs = self.identify_label(l_node, lst_l_transform)

            if all_docs:
                doc = all_docs[0]
                if doc["filepaths"]:
                    # update l_node
                    l_node.obj_id = doc["_id"]
                    l_node.filepaths = doc["filepaths"]

                elif save_obtained:
                    label, l_transform = self.materialize_with_existing_doc(doc=doc, l_node=l_node)

                    # save obtained
                    l_transform.save_file(filepaths)
                    label.save_file(filepaths)

                    # update l_node
                    l_node.obj_id = doc["_id"]
                    l_node.filepaths = filepaths

                    # for return
                    label_obtained, l_trans_obtained = label, l_transform

                else:
                    # update l_node
                    l_node.obj_id = doc["_id"]

            elif save_obtained:
                label, l_transform = self.set_off_and_record(l_node, db)

                # save obtained
                l_transform.save_file(filepaths)
                label.save_file(filepaths)

                # update l_node
                l_node.obj_id = label.obj_id
                l_node.filepaths = label.filepaths

                # for return
                label_obtained, l_trans_obtained = label, l_transform

            else:
                label, l_transform = self.set_off_and_record(l_node, db)

                # update l_node
                l_node.obj_id = label.obj_id

                # for return
                label_obtained, l_trans_obtained = label, l_transform
        else:
            dh = DbHandler()
            doc = dh.search_by_obj_id(obj_id=l_node.obj_id, element="Label", db=db)

            if doc["filepaths"]:
                doc_filepaths = doc["filepaths"]    # Prevent potentials errors resulted from different
                                                    # filepaths from doc and pipe_init

                # update l_node
                if l_node.filepaths is None:
                    l_node.filepaths = doc_filepaths

            elif save_obtained:
                label, l_transform = self.materialize_with_existing_doc(doc=doc, l_node=l_node)

                # save obtained
                label.save_file(filepaths)
                l_transform.save_file(filepaths)

                # update l_node
                l_node.filepaths = filepaths

                # for return
                label_obtained, l_trans_obtained = label, l_transform

            else:
                label, l_transform = self.materialize_with_existing_doc(doc=doc, l_node=l_node)

                # for return
                label_obtained, l_trans_obtained = label, l_transform

        return label_obtained, l_trans_obtained

    def materialize_with_existing_doc(self, doc, l_node):
        """
        From the document we found, recover the Label and the LTransform object.
        This should be used when a record is found in the db but the object itself is not saved

        :param doc:
        :param l_node:
        :return:
        """
        db = l_node.pipe_init.db
        frame = l_node.pipe_init.frame
        lab_fed = l_node.lab_fed.obj_id
        l_trans_id = doc["essentials"]["l_transform"]

        mp = MiniPipe()
        l_values, l_transform = mp.flow_to(l_node)

        l_transform.obj_id = l_trans_id
        l_transform.set_db(db)
        self.matched.append(l_trans_id)

        label = Label(frame=frame, l_transform=l_transform.obj_id, raw_y=lab_fed, values=l_values)
        label.set_db(db)
        label.obj_id = doc["_id"]

        return label, l_transform

    def set_off_and_record(self, l_node, db):
        """
        Build the obect according to the "DNA" in l_node.
        This should be used when no record of the target object is found from the db

        :param l_node:
        :param db:
        :return:
        """

        frame = l_node.pipe_init.frame
        lab_fed = l_node.lab_fed.obj_id

        mp = MiniPipe()
        l_values, l_transform = mp.flow_to(l_node)
        l_transform.save_db(db)
        self.matched.append(l_transform.obj_id)

        label = Label(frame=frame, l_transform=l_transform.obj_id, raw_y=lab_fed, values=l_values)
        label.save_db(db)

        return label, l_transform

    @staticmethod
    def identify_label(l_node, lst_l_transform):
        frame = l_node.pipe_init.frame
        lab_fed = l_node.lab_fed.obj_id

        dh = DbHandler()
        all_docs = []
        for l_tran in lst_l_transform:
            tmp = Label(frame=frame, l_transform=l_tran, raw_y=lab_fed, values=None)
            all_docs.extend(dh.search_by_essentials(tmp, l_node.pipe_init.db))
        all_docs = sorted(all_docs, key=lambda d: not bool(d["filepaths"]))

        return all_docs

    def l_prepare_locate(self, l_node):
        if not l_node.lab_fed.obj_id:
            self.l_locate(l_node.lab_fed)

        dh = DbHandler()
        lst_transform_ids = dh.search_by_essentials(l_node.l_transform, l_node.pipe_init.db)
        lst_transform_ids = [x["_id"]for x in lst_transform_ids if x["_id"] not in self.matched]
        return lst_transform_ids


class FConnector(object):
    def __init__(self, matched):
        self.matched = matched["f"]
        self.l_matched = matched["l"]

    def f_locate(self, f_node, save_obtained=True):
        if not isinstance(f_node, FNode):
            raise TypeError("The parameter f_node should of the type FNode.")

        feature_obtained, f_trans_obtained = None, None
        db = f_node.pipe_init.db
        filepaths = f_node.pipe_init.filepaths

        if f_node.obj_id is None:
            if not isinstance(f_node.l_node, LNode):
                raise TypeError("The attribute f_node.l_node should be of the type LNode")

            lst_f_transform = self.f_prepare_locate(f_node)
            all_docs = self.identify_feature(f_node, lst_f_transform)

            if all_docs:
                doc = all_docs[0]
                if doc["filepaths"]:
                    # update f_node
                    f_node.obj_id = doc["_id"]
                    f_node.filepaths = doc["filepaths"]

                elif save_obtained:
                    feature, f_transform = self.materialize_with_existing_doc(f_node=f_node, doc=doc)

                    # save obtained
                    f_transform.save_file(filepaths)
                    feature.save_file(filepaths)

                    # update f_node
                    f_node.obj_id = doc["_id"]
                    f_node.filepaths = filepaths

                    # for return
                    feature_obtained, f_trans_obtained = feature, f_transform

                else:
                    # update f_node
                    f_node.obj_id = doc["_id"]

            elif save_obtained:
                feature, f_transform = self.set_off_and_record(f_node, db)

                # save obtained
                f_transform.save_file(filepaths)
                feature.save_file(filepaths)

                # update f_node
                f_node.obj_id = feature.obj_id
                f_node.filepaths = feature.filepaths

                # for return
                feature_obtained, f_trans_obtained = feature, f_transform

            else:
                feature, f_transform = self.set_off_and_record(f_node, db)

                # update f_node
                f_node.obj_id = feature.obj_id

                # for return
                feature_obtained, f_trans_obtained = feature, f_transform

        else:
            dh = DbHandler()
            doc = dh.search_by_obj_id(obj_id=f_node.obj_id, element="Feature", db=db)

            if doc["filepaths"]:
                doc_filepaths = doc["filepaths"]    # Prevent potentials errors resulted from different
                                                    # filepaths from doc and pipe_init

                # update f_node
                if f_node.filepaths is None:
                    f_node.filepaths = filepaths

                # TODO: we should probably remove this part since nothing is "obtained" here
                # ih = IOHandler()
                # feature = ih.load_obj_from_file(f_node.obj_id, "Feature", doc_filepaths)
                # f_transform = ih.load_obj_from_file(doc["essentials"]["f_transform"], "FTransform", doc_filepaths)
                #
                # # for return
                # feature_obtained, f_trans_obtained = feature, f_transform

            elif save_obtained:
                feature, f_transform = self.materialize_with_existing_doc(f_node=f_node, doc=doc)

                # save obtained
                f_transform.save_file(filepaths)
                feature.save_file(filepaths)

                # update f_node
                f_node.filepaths = filepaths

                # for return
                feature_obtained, f_trans_obtained = feature, f_transform
            else:
                feature, f_transform = self.materialize_with_existing_doc(f_node=f_node, doc=doc)

                # for return
                feature_obtained, f_trans_obtained = feature, f_transform

        return feature_obtained, f_trans_obtained

    @staticmethod
    def identify_feature(f_node, lst_f_transform):
        frame = f_node.pipe_init.frame
        lst_fed = [f.obj_id for f in f_node.lst_fed]

        dh = DbHandler()
        all_docs = []
        for f_tran in lst_f_transform:
            tmp = Feature(frame=frame, f_transform=f_tran, lst_fed=lst_fed, label=f_node.l_node.obj_id, values=None)
            all_docs.extend(dh.search_by_essentials(tmp, f_node.pipe_init.db))
        all_docs = sorted(all_docs, key=lambda d: not bool(d["filepaths"]))

        return all_docs

    def materialize_with_existing_doc(self, f_node, doc):
        """

        :param f_node:
        :param doc:
        :return:
        """
        db = f_node.pipe_init.db
        frame = f_node.pipe_init.frame
        label = doc["essentials"]["label"]
        lst_fed = [f.obj_id for f in f_node.lst_fed]
        f_trans_id = doc["essentials"]["f_transform"]

        mp = MiniPipe()
        f_values, f_transform, stage = mp.flow_to(f_node)

        f_transform.obj_id = f_trans_id
        f_transform.set_db(db)
        self.matched.append(f_trans_id)

        feature = Feature(
            frame=frame, f_transform=f_trans_id, label=label, lst_fed=lst_fed, values=f_values
        )
        feature.stage = stage
        feature.obj_id = doc["_id"]
        feature.set_db(db)

        return feature, f_transform

    def set_off_and_record(self, f_node, db):
        frame = f_node.pipe_init.frame
        lst_fed = [f.obj_id for f in f_node.lst_fed]
        label = f_node.l_node.obj_id

        mp = MiniPipe()
        f_values, f_transform, stage = mp.flow_to(f_node)

        f_transform.save_db(db)
        self.matched.append(f_transform.obj_id)

        feature = Feature(
            frame=frame, lst_fed=lst_fed, f_transform=f_transform.obj_id, label=label, values=f_values
        )
        feature.stage = stage
        feature.save_db(db)

        return feature, f_transform

    def f_prepare_locate(self, f_node):
        for node in f_node.lst_fed:
            if node.obj_id is None:
                self.f_locate(node)

        lc = LConnector(self.l_matched)
        lc.l_locate(f_node.l_node)

        dh = DbHandler()
        lst_transform_ids = dh.search_by_essentials(f_node.f_transform, f_node.pipe_init.db)
        lst_transform_ids = [x["_id"]for x in lst_transform_ids if x["_id"] not in self.matched]

        return lst_transform_ids
