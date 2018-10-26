from ml_forest.core.constructions.io_handler import IOHandler
from ml_forest.core.constructions.db_handler import DbHandler

from ml_forest.pipeline.pipe_init import PipeInit
from ml_forest.pipeline.links.knitor import Knitor


class OutlierRmPipeInit(PipeInit):
    ih = IOHandler()

    def __init__(
            self, ref_fnode, threshold=0.5, id_col_name="Id", save_ref=True,

            data=None, col_y=None, lst_layers=None, shuffle=False, stratified=False,
            col_selected=None, tag=None, db=None, filepaths=None
    ):
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
        toberemoved = set(ref_identifications[boo, ].ravel())

        data = data.loc[
            data[id_col_name].apply(lambda i: i not in toberemoved)
        ]

        tobeupdated = {"ref_id": ref_core.obj_id, "ref_feature_id": None, "threshold": threshold}
        if ref_fnode.obj_id:
            tobeupdated["ref_feature_id"] = ref_fnode.obj_id

        super(OutlierRmPipeInit, self).__init__(
            filepaths=filepaths, db=db, tag=tag, col_selected=col_selected,
            stratified=stratified, shuffle=shuffle, lst_layers=lst_layers,
            col_y=col_y, data=data)

        DbHandler.insert_tag(self.core, {"ref": tobeupdated})
