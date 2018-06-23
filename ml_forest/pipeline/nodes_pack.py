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
        pass

    def locate(self, node):
        if node.obj_id is None:
            lst_l_transform = self.prepare_locate(node)
            frame = node.pipe_init.frame
            lab_fed = node.lab_fed

            dh = DbHandler()
            for l_tran in lst_l_transform:
                tmp = Label(frame=frame, l_transform=l_tran, raw_y=lab_fed, values=None)
                tmp = dh.search_by_essentials(tmp, node.pipe_init.db)

                if bool(tmp):
                    node.obj_id = tmp
                    break

        if node.obj_id is None:
            mp = MiniPipe()
            mp.flow_to(node)

    def prepare_locate(self, node):
        if not node.lab_fed.obj_id:
            self.locate(node.lab_fed)

        dh = DbHandler()
        lst_transform_ids = dh.search_by_essentials(node.l_transform, node.pipe_init.db)
        lst_transform_ids = map(lambda x: x['_id'], lst_transform_ids)
        return lst_transform_ids
