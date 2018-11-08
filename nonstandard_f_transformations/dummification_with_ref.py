import numpy as np

from ml_forest.core.elements.ftrans_base import FTransform
from ml_forest.core.constructions.io_handler import IOHandler

from ml_forest.pipeline.links.knitor import Knitor


class DummyWithRef(FTransform):
    def __init__(self, ref_fnode, save_ref=True, rise=0):
        super(DummyWithRef, self).__init__(rise=rise)
        self.__ref_id = ref_fnode.core.obj_id

        kn = Knitor()
        if save_ref:
            kn.f_subknit(ref_fnode)
        _, ft = kn.f_knit(ref_fnode)
        self.ft = ft

        ref_core = ref_fnode.core
        self.__essentials = {"ref_id": ref_core.obj_id, "ref_feature_id": None, "ref_ft_id": None}
        if ref_fnode.obj_id:
            self.__essentials["ref_feature_id"] = ref_fnode.obj_id
        if ft.obj_id:
            self.__essentials["ref_ft_id"] = ft.obj_id

    def transform_with_ref(self, fnode):
        ih = IOHandler()
        core = fnode.core
        lst_fed = fnode.lst_fed

        fed_values = []
        stage = 0
        for fed_node in lst_fed:
            f = ih.load_obj_from_file(fed_node.obj_id, "Feature", core.filepaths)
            if f.stage > stage:
                stage = f.stage
            fed_values.append(f.values)

        stage = stage + self.rise

        fed_values = np.concatenate(fed_values, axis=1)
        values = self.ft.transform(fed_values)

        return values, stage

    def transform(self, fed_test_values):
        values = self.ft.transform(fed_test_values)

        return values

    @property
    def ref_id(self):
        return self.__ref_id
