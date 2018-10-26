from ml_forest.core.elements.ltrans_base import LTransform
from ml_forest.core.constructions.io_handler import IOHandler

from ml_forest.pipeline.links.knitor import Knitor


class RmOutlierRefCookDistance(LTransform):
    ih = IOHandler()

    def __init__(self, ref_fnode, threshold=0.5, id_col_name="Id", save_ref=True):
        super(RmOutlierRefCookDistance, self).__init__()
        self.__ref_id = ref_fnode.core.obj_id
        self.__threshold = threshold

        # find observations to be removed
        kn = Knitor()
        f, ft = kn.f_knit(ref_fnode)
        if save_ref and not ref_fnode.filepaths:
            f.save_file(ref_fnode.core.filepaths)
            ft.save_file(ref_fnode.core.filepaths)
        boo = f.values > threshold

        # find the identifications of the observations to be removed
        ref_core = ref_fnode.core
        ref_identifications = self.ih.load_obj_from_file(
            ref_core.init_features[id_col_name], "Feature", ref_core.filepaths
        ).values
        self.__toberemoved = set(ref_identifications[boo, ].ravel())

        self.__essentials = {"ref_id": ref_core.obj_id, "ref_feature_id": None, "threshold": self.threshold}
        if ref_fnode.obj_id:
            self.__essentials["ref_feature_id"] = ref_fnode.obj_id

    def transform_with_ref(self, lnode, id_col_name="Id"):
        core = lnode.core
        lab_fed = lnode.lab_fed

        identifications = self.ih.load_obj_from_file(
            core.init_features[id_col_name], "Feature", core.filepaths
        ).values.ravel()
        boo = [i not in self.__toberemoved for i in identifications]

        fed_values = self.ih.load_obj_from_file(lab_fed.obj_id, "Label", core.filepaths).values
        values = fed_values[boo, :]

        return values

    def transform(self, fed_test_value):
        msg = "One cannot decide outliers for a test dataset."

        raise ValueError(msg)

    @property
    def ref_id(self):
        return self.__ref_id

    @property
    def threshold(self):
        return self.__threshold
