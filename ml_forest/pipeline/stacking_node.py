from warnings import warn

from ml_forest.core.elements.ftrans_base import FTransform
from ml_forest.core.elements.ltrans_base import LTransform
from ml_forest.core.constructions.db_handler import DbHandler

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
        else:
            raise TypeError("Why do you need to update obj_id of a node?")

    # TODO: Decide if we want to add db or remove filepaths
    @property
    def filepaths(self):
        return self.__filepaths

    @filepaths.setter
    def filepaths(self, _paths):
        if not bool(self.filepaths):
            self.__filepaths = _paths
        else:
            raise TypeError("why do you need to update filepaths of a node?")


class FNode(StackingNode):
    def __init__(self, pipe_init, lst_fed=None, f_transform=None, l_node=None, obj_id=None):
        """

        :param pipe_init: PipeInit
        :param lst_fed:  a list of FNode or None
        :param f_transform: FTransform or None
        :param l_node: LNode or None
        :param obj_id: ObjectId or None. FNode can be created with obj_id provided directly
        """
        super(FNode, self).__init__()

        if obj_id:
            self.obj_id = obj_id
            if lst_fed or f_transform or l_node:
                warn("Because obj_id is provided, it's used. The other parameters are ignored.")
        else:
            # TODO: assert non-missing conditions (low priority)
            self.__creating_doc(pipe_init, lst_fed, f_transform, l_node)

    def __creating_doc(self, pipe_init, lst_fed, f_transform, l_node):
        if not isinstance(pipe_init, PipeInit):
            raise TypeError("The parameter pipe_init should be of the type ml_forest.pipe_init.PipeInit")
        self.pipe_init = pipe_init

        if lst_fed:
            for fed in lst_fed:
                if not isinstance(fed, FNode):
                    raise TypeError("Every element in the parameter lst_fed should be a FNode")
        self.lst_fed = lst_fed

        if f_transform and not isinstance(f_transform, FTransform):
            raise TypeError("The paramenter f_transform should be of the type FTransform")
        self.f_transform = f_transform

        if l_node and not isinstance(l_node, LNode):
            raise TypeError("The parameter l_node should be of the type LNode")
        self.l_node = l_node


class LNode(StackingNode):
    def __init__(self, pipe_init, lab_fed=None, l_transform=None, obj_id=None):
        super(LNode, self).__init__()

        if obj_id:
            self.obj_id = obj_id
            if pipe_init or lab_fed or l_transform:
                warn("Because obj_id is provided, it's used. The other parameters are ignored.")
        else:
            # TODO: assert non-missing conditions (low priority)
            self.__creating_doc(pipe_init, lab_fed, l_transform)

    def __creating_doc(self, pipe_init, lab_fed, l_transform):
        if not isinstance(pipe_init, PipeInit):
            raise TypeError("The parameter pipe_init should be of the type ml_forest.pipe_init.PipeInit")
        self.pipe_init = pipe_init

        if lab_fed and isinstance(lab_fed, LNode):
                    raise TypeError("Every element in the parameter lst_fed should be a LNode")
        self.lab_fed = lab_fed

        if l_transform and not isinstance(l_transform, LTransform):
            raise TypeError("The paramenter l_transform should be of the type LTransform")
        self.l_transform = l_transform


