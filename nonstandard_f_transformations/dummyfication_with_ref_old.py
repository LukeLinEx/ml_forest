import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from ml_forest.core.elements.ftrans_base import FTransform
from ml_forest.core.constructions.io_handler import IOHandler

from ml_forest.pipeline.pipe_init import PipeInit
from ml_forest.pipeline.nodes.stacking_node import FNode
from ml_forest.pipeline.links.knitor import Knitor

from feature_transformations.encoding.simple_dummy import SimpleDummy


class DummyWithRef(FTransform):
    def __init__(self, ref_id, keep_missing=False):
        super(DummyWithRef, self).__init__(rise=0)
        self.__ref_id = ref_id
        self.__essentials = {"ref_id": ref_id, "keep_missing": keep_missing}
        self._col_encoded = None
        self._all_classes = None

    @property
    def ref_id(self):
        return self.__ref_id

    @property
    def col_encoded(self):
        return self._col_encoded

    @property
    def all_classes(self):
        return self._all_classes

    def transform_with_ref(self, fnode):
        filepaths = fnode.core.filepaths

        ref_init = PipeInit(pipe_id=self.ref_id, filepaths=filepaths)
        pipe_init = PipeInit(pipe_id=fnode.core.obj_id, filepaths=filepaths)
        ref_core = ref_init.core
        ref_transform = SimpleDummy()

        lst_fed_id = [f.obj_id for f in fnode.lst_fed]
        f_names = [
            key for key in pipe_init.init_fnodes if pipe_init.init_fnodes[key].obj_id in lst_fed_id
            ]
        ref_fed_fnodes = [ref_init.init_fnodes[fname] for fname in f_names]

        ref_node = FNode(core=ref_core, lst_fed=ref_fed_fnodes, f_transform=ref_transform)

        kn = Knitor()
        kn.f_subknit(ref_node)  # Force save?

        self.__essentials["ref_feature_id"] = ref_node.obj_id
        self.__essentials["ref_f_transform_id"] = ref_transform.obj_id
        self._col_encoded = ref_transform.col_encoded
        self._all_classes = ref_transform.all_classes

        ih = IOHandler()
        lst_fed = [ih.load_obj_from_file(f_id, "Feature", filepaths) for f_id in lst_fed_id]
        # TODO, reference: f_collect_components in minipipe
        prevstage = max(map(lambda x: x.stage, lst_fed))
        stage = prevstage

        if len(lst_fed) == 1:
            fed_values = lst_fed[0].values
        else:
            fed_values = np.concatenate(list(map(lambda x: x.values, lst_fed)), axis=1)

        dummy_values = ref_transform.transform(fed_values)

        return dummy_values, stage

    def transform(self, fed_test_value):
        if not bool(self._col_encoded):
            raise NotFittedError("The {} object is not fitted yet".format(str(type(self))))

        if len(fed_test_value.shape) == 2 and fed_test_value.shape[1] == 1:
            fed_test_value = fed_test_value.reshape(-1, )
        elif len(fed_test_value.shape) == 1:
            pass
        else:
            raise NotImplementedError("The test data don't have the same shape as in the training data")

        tmp = pd.Series(fed_test_value)

        tmp.loc[tmp.isnull()] = "This is a missing value"
        unseen = tmp.apply(lambda x: x not in self._all_classes)
        if np.sum(unseen) > 0:
            warnings.warn("Warning: some class in test data hasn't been seen in the training data")

        value = (fed_test_value.reshape(-1, 1) == np.array(self._col_encoded).reshape(1, -1)).astype(float)

        if self.essentials["keep_missing"]:
            value[unseen, :] = np.nan
        else:
            value[unseen, :] = 0

        return value
