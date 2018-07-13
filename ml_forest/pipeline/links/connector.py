from ml_forest.core.constructions.db_handler import DbHandler
from ml_forest.core.constructions.io_handler import IOHandler

from ml_forest.pipeline.nodes.stacking_node import LNode


class LConnector(object):
    def __init__(self, matched):
        self.matched = matched

    def get_l_transform_candidates(self, l_node):
        dh = DbHandler()
        lst_transform_ids = dh.search_by_essentials(l_node.l_transform, l_node.pipe_init.db)
        lst_transform_ids = [x["_id"]for x in lst_transform_ids if x["_id"] not in self.matched]
        return lst_transform_ids

    def materialize_with_existing_doc(self, doc, l_node):
        """
        From the document we found, recover the Label and the LTransform object.
        This should be used when a record is found in the db but the object itself is not saved

        :param doc:
        :param l_node:
        :return:
        """
        db = l_node.pipe_init.db

        frame_id = l_node.pipe_init.frame
        lab_fed = l_node.lab_fed.obj_id
        l_transform_id = doc["essentials"]["l_transform"]

        mp = MiniPipe()
        l_values, l_transform = mp.flow_to(l_node)

        l_transform.obj_id = l_trans_id
        l_transform.set_db(db)
        self.matched.append(l_trans_id)

        label = Label(frame=frame, l_transform=l_transform.obj_id, raw_y=lab_fed, values=l_values)
        label.set_db(db)
        label.obj_id = doc["_id"]

        return label, l_transform

    def materialize(self, l_node):
        """
        Assuming all the components in this l_node are ready. Generate the Label and the LTransform based on those.

        :param l_node: LNode
        :return:
        """
        if not isinstance(l_node, LNode):
            raise TypeError("The parameter l_node should be of the type LNode.")

        label_obtained, l_trans_obtained = None, None
        db = l_node.pipe_init.db
        filepaths = l_node.pipe_init.filepaths
        lst_l_transform = self.get_l_transform_candidates(l_node)
        all_docs = l_node.get_docs_match_the_lnode(lst_l_transform)

        if all_docs:
            doc = all_docs[0]
            if doc["filepaths"]:
                filepaths = doc["filepaths"]
                label_id = doc["_id"]
                l_transform_id = doc["l_transform"]

                ih = IOHandler()
                label = ih.load_obj_from_file(label_id, "Label", filepaths)
                l_transform = ih.load_obj_from_file(l_transform_id, "LTransform", filepaths)
            else:
                label, l_transform = self.materialize_with_existing_doc(doc, l_node)
        else: # not found
            pass



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