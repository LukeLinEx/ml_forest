"""
# TODO: this is a temperary version. We need a thorough refactor
Init:
    - save
    - extract obj
"""


class TestSetProcess(object):
    def __init__(self, test_id=None, test_data=None, core_int=None, tag=None):
        """
        Creating the starting point for the test data. We need to select the same columns, and identify them with
        the FeatureStarter objects.

        TestProcess can have label or not. If not, it's not testable, we can only predict.

        Through testing, (the obj_id of) new TestFeature would be added to the self.__test_features
        # TODO: try to update self.__test_features in db only; save only the actual features to storage

        No essentials for TestProcess. A test process is identifies by a test data frame, which is not
        part of what we are tracking.

        :param test_data: A pandas DataFrame of test data
        :param core_init: core.constructions.core_init.CoreInit object provides:
                - identities: the dictionary of {FeatureStarter.obj_id: [col_names]}.
                - col_y: the name for the column of the labels. Used for evaluation.
                - db: where self.__test_features should be updated to
                - filepaths: wherer the new test_features should be saved to
        """
        if test_id:
            """
            Recover TestSetProcess object from db and ignore other params passed
            """
            pass
        else:
            self.db = core_int.db
            self.filepaths = core_int.filepaths
            self.element = "TestSetProcess"

            """
            - create a document in db
            - initialize test_features
            - get label (for evaluation)
            """


