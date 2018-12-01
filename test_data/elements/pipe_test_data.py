from ml_forest.core.elements.identity import Base
from ml_forest.core.constructions.db_handler import DbHandler
from test_data.elements.test_feature import TestFeature


class PipeTestData(Base):
    def __init__(self, test_data=None, core=None, tag=None, db=None, filepaths=None):
        """
        Creating the starting point for the test data. We need to select the same columns, and identify them with
        the core.init_features.

        PipeTestData can have label or not. If not, it's not evaluable, we can only predict.

        Through predicting, (the obj_id of) new TestFeature would be added to the PipeTestData.__test_features

        Empty essentials for PipeTestData. A test process is identified by a test data frame, which is not
        part of what we are tracking.

        :param test_data: A pandas DataFrame of test data
        :param core: ml_forest.core.construction.core_init.CoreInit object provides:
                - _column_groups/col_selected: the dictionary of {feature_name: [col_names]}.
                - init_features: the dictionary of {feature_name: feature.obj_id}
                - col_y: the name for the column of the labels. Used for evaluation.
                - init_label.obj_id
        """
        if not db or not filepaths:
            db = core.db
            filepaths = core.filepaths

        super(PipeTestData, self).__init__()
        self.__essentials = {"core_init": core.obj_id}

        dh = DbHandler()
        self.test_features = {}

        self.save_db(db)
        self.__get_init_test_features(core, test_data)
        self.__label = PipeTestData.__get_label(core, test_data)

        dh.insert_tag(self, {"tag": tag})
        self.save_file(filepaths)
        print(self.obj_id)

    @staticmethod
    def decide_element():
        return "PipeTestData"

    def __get_init_test_features(self, core, test_data):
        for name in core.init_features:
            colnames = core._column_groups[name]
            fval = test_data[colnames].values

            test_f = TestFeature(self.obj_id, fval)
            test_f.save_db_file(core.db, core.filepaths)
            self.test_features[core.init_features[name]] = test_f.obj_id

    @staticmethod
    def __get_label(core, test_data):
        lval = None
        col_y = core._y_name
        if bool(col_y) and col_y in test_data.columns:
            lval = test_data[col_y]

        return lval

    @property
    def label(self):
        return self.__label
