import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ml_forest.core.elements.identity import Base
from ml_forest.core.elements.frame_base import Frame, FrameWithDeepestLayerSpecified
from ml_forest.core.elements.feature_base import Feature
from ml_forest.core.elements.label_base import Label

from ml_forest.core.constructions.db_handler import DbHandler


# TODO: reverse index (low priority)

class PipeInit(Base):
    def __init__(self, data, col_y, lst_layers, shuffle=False, stratified=False, col_selected=None,
                 tag=None, db=None, filepaths=None):
        """

        :param data: pandas.DataFrame. This needs to be a pandas data frame with a label column
        :param col_y: The name of the label column
        :param lst_layers: list. This gives the "lst_layers" to the Frame
        :param shuffle: boolean.
        :param stratified: boolean. Should not be used to a regression problem
        :param col_selected: dict. Ex: {'num': ['colname1', 'colname2'], 'cate':['colname3'], ...}
        :param db:
        :param filepaths:
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("The data for initializing a pipe should be of the type pandas.DataFrame")
        if col_y not in data:
            raise KeyError("The column name of the target: col_y provided is not in the data")

        super(PipeInit, self).__init__(db=db, filepaths=filepaths)
        self.__essentials = {}

        # Initializing the rows
        if shuffle:
            idx = np.random.choice(data.index, len(data.index), replace=False)
            data = self.shuffle_pddf_idx(data, idx)

        if stratified:
            data, frame = self.get_stratified_starter_and_frame(lst_layers, data, col_y)
        else:
            frame = self.get_regular_frame(lst_layers, data)
        self.__frame = frame.obj_id

        # Initializing labels
        values = data[[col_y]].values
        label = Label(frame.obj_id, None, None, values, db=self.db, filepaths=self.filepaths)
        self.__label = label.obj_id

        # Initializing features (columns)
        self._column_groups = {}  # to collect dict like {'num': ['colname1', 'colname2'], 'cate':['colname3'], ...}
        self._init_features = {}  # {'num': obj_id(data['colname1', 'colname2']),
                                         #  'cate': obj_id(data['colname3']), ...}

        self._y_name = col_y
        if isinstance(col_selected, dict):
            for key in col_selected:
                cols = col_selected[key]
                self._column_groups[key] = cols

                values = data[cols].values
                feature = Feature(
                    frame.obj_id, None, None, None, values=values, db=self.db, filepaths=self.filepaths
                )
                self._init_features[key] = feature.obj_id
        elif not col_selected:
            cols = data.columns

            values = data[cols].values
            feature = Feature(
                frame.obj_id, None, None, None, values=values, db=self.db, filepaths=self.filepaths
            )
            self._init_features['raw'] = feature.obj_id
        elif isinstance(col_selected, list):
            raise NotImplementedError("Currently only support dictionary to initialize features")
        else:
            raise ValueError("Don't know what to do with the way you specified columns")

        if type(self) == PipeInit:
            self.save_db_file()

        DbHandler.insert_tag(self, {"tag": tag})

    @staticmethod
    def shuffle_pddf_idx(df, idx):
        return df.iloc[idx].reset_index(drop=True)

    def get_regular_frame(self, lst_layers, data):
        num_observations = data.shape[0]
        db = self.db
        filepaths = self.filepaths
        frame = Frame(num_observations, lst_layers, db=db, filepaths=filepaths)
        return frame

    def get_stratified_starter_and_frame(self, lst_layers, data, col_y):
        n_splits =1
        for i in lst_layers:
            n_splits *= i
        skf = StratifiedKFold(n_splits=n_splits)
        folds_deepest_layer = skf.split(range(data.shape[0]), data[col_y])

        folds = []
        while True:
            try:
                _, fold = folds_deepest_layer.next()
                folds.append(fold)
            except StopIteration:
                break

        len_folds_deepest_layer = map(len, folds)
        idx = np.concatenate(folds)
        X = self.shuffle_pddf_idx(data, idx)

        db = self.db
        filepaths = self.filepaths
        frame = FrameWithDeepestLayerSpecified(
            num_observations=len(idx), lst_layers=lst_layers,
            len_folds_deepest_layer=len_folds_deepest_layer,
            db=db, filepaths=filepaths
        )

        return X, frame

if __name__ == "__main__":
    print(Label)