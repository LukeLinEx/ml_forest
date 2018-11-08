import numpy as np

from ml_forest.core.elements.ftrans_base import FTransform
from ml_forest.core.constructions.io_handler import IOHandler

from ml_forest.pipeline.links.knitor import Knitor


class FeatureSelectionLasso(FTransform):
    ih = IOHandler()

    def __init__(self, ref_fnode, save_ref=True, rise=0):
        super(FeatureSelectionLasso, self).__init__(rise=rise)
        self.__ref_id = ref_fnode.core.obj_id

        kn = Knitor()
        f, ft = kn.f_knit(ref_fnode)
        if save_ref and not ref_fnode.filepaths:
            f.save_file(ref_fnode.core.filepaths)
            ft.save_file(ref_fnode.core.filepaths)

        coefs_ = np.concatenate([mod.coef_.reshape(1, -1) for mod in ft.models.values()], axis=0)
        majority_vote = np.mean(coefs_ > 0, axis=0)
        self.__tobe_kept = majority_vote >= 0.5

        ref_core = ref_fnode.core
        self.__essentials = {"ref_id": ref_core.obj_id, "ref_feature_id": None}
        if ref_fnode.obj_id:
            self.__essentials["ref_feature_id"] = ref_fnode.obj_id

    def transform_with_ref(self, fnode):
        core = fnode.core
        lst_fed = fnode.lst_fed

        fed_values = []
        stage = 0
        for fed_node in lst_fed:
            f = self.ih.load_obj_from_file(fed_node.obj_id, "Feature", core.filepaths)
            if f.stage > stage:
                stage = f.stage
            fed_values.append(f.values)

        stage = stage + self.rise

        fed_values = np.concatenate(fed_values, axis=1)
        values = fed_values[:, self.__tobe_kept]

        return values, stage

    def transform(self, fed_test_value):
        values = fed_test_value[:, self.__tobe_kept]

        return values

    @property
    def ref_id(self):
        return self.__ref_id
