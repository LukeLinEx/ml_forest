import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

from ml_forest.core.elements.ftrans_base import FTransform
from ml_forest.core.constructions.io_handler import IOHandler

from ml_forest.pipeline.pipe_init import PipeInit
from ml_forest.pipeline.nodes.stacking_node import LNode
from ml_forest.pipeline.links.knitor import Knitor

from label_transformations.log_transform import LogTransform


class CookDistWithRef(FTransform):
    def __init__(self, ref_id, threshold=0.05):
        super(CookDistWithRef, self).__init__(rise=0)
        self.__ref_id = ref_id
        self.__threshold = threshold

        self.__essentials = {"ref_id": self.__ref_id}

    @property
    def ref_id(self):
        return self.__ref_id

    def __colname_tmp(self, n):
        return "f_" + str(n) if str(n) != "y" else "y"

    def transform_for_ref(self, fnode):
        filepaths = fnode.core.filepaths
        pipe_init = PipeInit(pipe_id=fnode.core.obj_id, filepaths=filepaths)

        ref_init = PipeInit(pipe_id=self.ref_id, filepaths=filepaths)
        ref_core = ref_init.core

        lst_fed_id = [f.obj_id for f in fnode.lst_fed]
        f_names = [key for key in pipe_init.init_fnodes if pipe_init.init_fnodes[key].obj_id in lst_fed_id]

        ref_lst_fed_id = []
        for fname in f_names:
            try:
                node = ref_init.init_fnodes[fname]
                ref_fed_obj_id = node.obj_id
                ref_lst_fed_id.append(ref_fed_obj_id)
            except KeyError:
                raise KeyError("The feature {} can't be found among the initial features.".format(fname))
            except AttributeError:
                raise AttributeError("The feature {} doesn't have the object id.".format(fname))

        ih = IOHandler()
        feature_values = []
        for f_id in ref_lst_fed_id:
            feature = ih.load_obj_from_file(f_id, "Feature", filepaths)
            feature_values.append(feature.values)
        _id = ih.load_obj_from_file(
            ref_core.init_features["Id"], "Feature", filepaths
        )

        feature_values = np.concatenate(feature_values, axis=1)

        kn = Knitor()
        lnode = LNode(ref_core, ref_init.init_lnode, LogTransform(shift=1))
        l, lt = kn.l_knit(lnode)

        # Modeling part
        data = pd.DataFrame(feature_values)
        data.rename(columns=self.__colname_tmp, inplace=True)
        data["y"] = l.values.ravel()

        rhs = "+".join(filter(lambda s: s != "y", data.columns))

        model = sm.ols(formula='y~{}'.format(rhs), data=data)
        fitted = model.fit()
        influence = fitted.get_influence()
        (c, p) = influence.cooks_distance

        ref_cd = pd.DataFrame({"_id": _id, "cd": c}).sort_values("cd", ascending=False)
        outlier_id = set(ref_cd.loc[ref_cd["cd"] > self.__threshold, "_id"])

        # Filter the outliers
        col_ids = ih.load_obj_from_file(
            pipe_init.init_features["Id"], "Feature", filepaths
        )

        col_ids = [x in outlier_id for x in col_ids.values.ravel()]

        feature_values = []
        for f in fnode.lst_fed:
            f_id = f.obj_id
            feature = ih.load_obj_from_file(f_id, "Feature", filepaths)
            feature_values.append(feature.values)
        feature_values = np.concatenate(feature_values, axis=1)

        final = feature_values[col_ids, :]

        return final
