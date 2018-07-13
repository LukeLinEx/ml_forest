from warnings import warn

from ml_forest.core.elements.feature_base import Feature
from ml_forest.core.elements.label_base import Label
from ml_forest.core.elements.ftrans_base import FTransform
from ml_forest.core.elements.ltrans_base import LTransform
from ml_forest.core.constructions.db_handler import DbHandler
from ml_forest.core.constructions.io_handler import IOHandler

from ml_forest.pipeline.pipe_init import PipeInit


class StackingNode(object):
    def __init__(self):
        self.__obj_id = None
        self.__filepaths = None

    @property
    def obj_id(self):
        return self.__obj_id

    @obj_id.setter
    def obj_id(self, _id):
        if not bool(self.obj_id):
            self.__obj_id = _id
        elif _id == self.obj_id:
            pass
        else:
            raise TypeError("Why do you need to update obj_id of a node?")

    @property
    def filepaths(self):
        return self.__filepaths

    @filepaths.setter
    def filepaths(self, _paths):
        if not bool(self.filepaths):
            self.__filepaths = _paths
        else:
            raise TypeError("This method is not designed to update filepaths.")

    def fetch(self):
        if self.obj_id is None or self.filepaths is None:
            msg = "The node doesn't have obj_id yet. The function is designed to fetch an obj whose location is " +\
                  "specified in a node."
            raise ValueError(msg)

        obj_id = self.obj_id
        element = self.decide_element()
        filepaths = self.pipe_init.filepaths

        ih = IOHandler()
        obj_fetched = ih.load_obj_from_file(obj_id, element, filepaths)

        return obj_fetched

    @staticmethod
    def decide_element():
        raise NotImplementedError

    @property
    def pipe_init(self):
        raise NotImplementedError


class FNode(StackingNode):
    def __init__(self, pipe_init, lst_fed=None, f_transform=None, l_node=None, obj_id=None, filepaths=None):
        """

        :param pipe_init: PipeInit
        :param lst_fed:  a list of FNode or None
        :param f_transform: FTransform or None
        :param l_node: LNode or None
        :param obj_id: ObjectId or None. FNode can be created with obj_id provided directly
        :param filepaths: list of dict. This indicates that the object is saved in one of the paths in the list.
                          This will only be used if obj_id is provided.
        """
        super(FNode, self).__init__()

        if obj_id:
            self.obj_id = obj_id
            if filepaths:
                self.filepaths = filepaths
            if lst_fed or f_transform or l_node:
                warn("Because obj_id is provided, it's used. The other parameters are ignored.")
        else:
            # TODO: assert non-missing conditions (low priority)
            pipe_init, lst_fed, f_transform, l_node = self.__inspect_doc(pipe_init, lst_fed, f_transform, l_node)
            self._pipe_init = pipe_init
            self.lst_fed = lst_fed
            self.f_transform = f_transform
            self.l_node = l_node

    def __inspect_doc(self, pipe_init, lst_fed, f_transform, l_node):
        if not isinstance(pipe_init, PipeInit):
            raise TypeError("The parameter pipe_init should be of the type ml_forest.pipe_init.PipeInit")

        if lst_fed:
            for fed in lst_fed:
                if not isinstance(fed, FNode):
                    raise TypeError("Every element in the parameter lst_fed should be a FNode")

        if f_transform and not isinstance(f_transform, FTransform):
            raise TypeError("The paramenter f_transform should be of the type FTransform")

        if l_node and not isinstance(l_node, LNode):
            raise TypeError("The parameter l_node should be of the type LNode")

        return pipe_init, lst_fed, f_transform, l_node

    def get_docs_match_the_fnode(self, lst_f_transform):
        frame = self.pipe_init.frame
        lst_fed = [f.obj_id for f in self.lst_fed]

        dh = DbHandler()
        all_docs = []
        for f_tran in lst_f_transform:
            tmp = Feature(frame=frame, f_transform=f_tran, lst_fed=lst_fed, label=self.l_node.obj_id, values=None)
            all_docs.extend(dh.search_by_essentials(tmp, self.pipe_init.db))
        all_docs = sorted(all_docs, key=lambda d: not bool(d["filepaths"]))

        return all_docs

    @staticmethod
    def decide_element():
        return "Feature"

    @property
    def pipe_init(self):
        return self._pipe_init


class LNode(StackingNode):
    def __init__(self, pipe_init, lab_fed=None, l_transform=None, obj_id=None, filepaths=None):
        """

        :param pipe_init:
        :param lab_fed:
        :param l_transform:
        :param obj_id:
        :param filepaths: list of dict. This indicates that the object is saved in one of the paths in the list.
                          This will only be used if obj_id is provided.
        """
        super(LNode, self).__init__()

        if obj_id:
            self.obj_id = obj_id
            if filepaths:
                self.filepaths = filepaths
            if pipe_init or lab_fed or l_transform:
                warn("Because obj_id is provided, it's used. The other parameters are ignored.")
        else:
            # TODO: assert non-missing conditions (low priority)
            pipe_init, lab_fed, l_transform = self.__inspect_doc(pipe_init, lab_fed, l_transform)
            self._pipe_init = pipe_init
            self.lab_fed = lab_fed
            self.l_transform = l_transform

    def __inspect_doc(self, pipe_init, lab_fed, l_transform):
        if not isinstance(pipe_init, PipeInit):
            raise TypeError("The parameter pipe_init should be of the type ml_forest.pipe_init.PipeInit")

        if lab_fed and not isinstance(lab_fed, LNode):
            raise TypeError("Every element in the parameter lab_fed should be a LNode")

        if l_transform and not isinstance(l_transform, LTransform):
            raise TypeError("The paramenter l_transform should be of the type LTransform")

        return pipe_init, lab_fed, l_transform

    def get_docs_match_the_lnode(self, lst_l_transform):
        frame = self.pipe_init.frame
        lab_fed = self.lab_fed.obj_id

        dh = DbHandler()
        all_docs = []
        for l_tran in lst_l_transform:
            tmp = Label(frame=frame, l_transform=l_tran, raw_y=lab_fed, values=None)
            all_docs.extend(dh.search_by_essentials(tmp, self.pipe_init.db))
        all_docs = sorted(all_docs, key=lambda d: not bool(d["filepaths"]))

        return all_docs

    @staticmethod
    def decide_element():
        return "Label"

    @property
    def pipe_init(self):
        return self._pipe_init
