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

# TODO: (HIGH PRIORITY) Implement `materialized` in locate. If materialized==true, test saved and save when created
# TODO: (HIGH PRIORITY) Implement with a better logic
# TODO: inspect whether a obj_id should be used
class Connector(object):
    def __init__(self):
        self.__matched = {
            "l_transform": [],
            "f_transform": []
        }

    def l_locate(self, l_node, save_new_obtained=True):
        if not isinstance(l_node, LNode):
            raise TypeError("The parameter l_node should be of the type LNode.")

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
            label, l_transform = mp.flow_to(l_node)
            l_node.obj_id = label.obj_id
            if save_new_obtained:
                l_transform.set_filepaths(l_node.pipe_init.filepaths)
                l_transform.save_file()
                label.set_filepaths(l_node.pipe_init.filepaths)
                label.save_file()
            else:
                return label, l_transform

    def l_prepare_locate(self, l_node):
        if not l_node.lab_fed.obj_id:
            self.l_locate(l_node.lab_fed)

        dh = DbHandler()
        lst_transform_ids = dh.search_by_essentials(l_node.l_transform, l_node.pipe_init.db)
        lst_transform_ids = [x["_id"]for x in lst_transform_ids if x["_id"] not in self.__matched["l_transform"]]
        return lst_transform_ids

    def f_locate(self, f_node, save_new_obtained=True):
        if not isinstance(f_node, FNode):
            raise TypeError("The parameter f_node should of the type FNode.")

        if f_node.obj_id is None:
            if not isinstance(f_node.l_node, LNode):
                raise TypeError("The label of the f_node should be of the type LNode")
            lst_transform_ids = self.f_prepare_locate(f_node)
            frame = f_node.pipe_init.frame
            lst_fed = [f.obj_id for f in f_node.lst_fed]
            label = f_node.l_node.obj_id

            dh = DbHandler()
            for f_tran in lst_transform_ids:
                tmp = Feature(frame=frame, lst_fed=lst_fed, label=label, f_transform=f_tran, values=None)
                tmp = dh.search_by_essentials(tmp, f_node.pipe_init.db)

                if bool(tmp):
                    f_node.obj_id = tmp
                    self.__matched["f_transform"].append(tmp)
                    break

        # train if FNode still has no obj_id after searching
        if f_node.obj_id is None:
            mp = MiniPipe()
            feature, f_transform = mp.flow_to(f_node)
            if save_new_obtained:
                f_transform.set_filepaths(f_node.pipe_init.filepaths)
                f_transform.save_file()
                feature.set_filepaths(f_node.pipe_init.filepaths)
                feature.save_file()
            else:
                return feature, f_transform

    def f_prepare_locate(self, f_node):
        for node in f_node.lst_fed:
            if node.obj_id is None:
                self.f_locate(node)

        self.l_locate(f_node.l_node)

        dh = DbHandler()
        lst_transform_ids = dh.search_by_essentials(f_node.f_transform, f_node.pipe_init.db)
        lst_transform_ids = [x["_id"]for x in lst_transform_ids if x["_id"] not in self.__matched["f_transform"]]

        return lst_transform_ids
