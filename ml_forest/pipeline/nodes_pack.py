from ml_forest.core.elements.feature_base import Feature
from ml_forest.core.elements.label_base import Label
from ml_forest.core.constructions.db_handler import DbHandler
from ml_forest.core.constructions.mini_pipe import MiniPipe

from ml_forest.pipeline.stacking_node import FNode, LNode


class Assembler(object):
    def __init__(self):
        self.nodes = []

    def collect_and_sort(self, obj_id_dict):
        obj_id_prlst = sorted(obj_id_dict.items(), key=lambda p: p[0])
        obj_id_lst = [p[1] for p in obj_id_prlst]
        return obj_id_lst

    def get_nodes(self, pipe_init, obj_id_dict):
        obj_id_lst = self.collect_and_sort(obj_id_dict)
        nodes = [FNode(pipe_init=pipe_init, obj_id=oid) for oid in obj_id_lst]
        self.nodes = nodes


class Connector(object):
    def __init__(self):
        self.__matched = {
            "l_transform": [],
            "f_transform": []
        }

    def l_locate(self, l_node):
        if l_node.obj_id is None:
            lst_l_transform = self.l_prepare_locate(l_node)
            frame = l_node.pipe_init.frame
            lab_fed = l_node.lab_fed

            dh = DbHandler()
            for l_tran in lst_l_transform:
                tmp = Label(frame=frame, l_transform=l_tran, raw_y=lab_fed, values=None)
                tmp = dh.search_by_essentials(tmp, l_node.pipe_init.db)

                if bool(tmp):
                    l_node.obj_id = tmp
                    self.__matched["l_transform"].append(l_tran)
                    break

        if l_node.obj_id is None:
            mp = MiniPipe()
            mp.flow_to(l_node)

    def l_prepare_locate(self, l_node):
        if not l_node.lab_fed.obj_id:
            self.l_locate(l_node.lab_fed)

        dh = DbHandler()
        lst_transform_ids = dh.search_by_essentials(l_node.l_transform, l_node.pipe_init.db)
        lst_transform_ids = [x["_id"]for x in lst_transform_ids if x["_id"] not in self.__matched["l_transform"]]
        return lst_transform_ids

    def f_locate(self, f_node):
        if f_node.obj_id is None:
            if not isinstance(f_node.label, LNode):
                raise TypeError("The label of the f_node should be of the type LNode")
            lst_transform_ids = self.f_prepare_locate(f_node)
            frame = f_node.pipe_init.frame
            lst_fed = f_node.lst_fed
            label = f_node.label

            dh = DbHandler()
            for f_tran in lst_transform_ids:
                tmp = Feature(frame=frame, lst_fed=lst_fed, label=label, f_transformer=f_tran, values=None)
                tmp = dh.search_by_essentials(tmp, f_node.pipe_init.db)

                if bool(tmp):
                    f_node.obj_id = tmp
                    self.__matched["f_transform"].append(tmp)
                    break

        # train if FNode still has no obj_id afer searching
        if f_node.obj_id is None:
            mp = MiniPipe()
            mp.flow_to(f_node)

    def f_prepare_locate(self, f_node):
        for node in f_node.lst_fed:
            if node.obj_id is None:
                self.f_locate(node)

        self.l_locate(f_node.label)

        dh = DbHandler()
        lst_transform_ids = dh.search_by_essentials(f_node.f_transform, f_node.pipe_init.db)
        lst_transform_ids = [x["_id"]for x in lst_transform_ids if x["_id"] not in self.__matched["f_transform"]]

        return lst_transform_ids
