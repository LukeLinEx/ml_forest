from ml_forest.core.elements.identity import Base


class TestFeature(Base):
    def __init__(self, pipe_test, values):
        """

        :param pipe_test: obj_id of the PipeTestData object
        :param values: 2-dimentional numpy array
        """
        super(TestFeature, self).__init__()

        self.__essentials = {"pipe_test": pipe_test}
        self.__values = values

    @property
    def values(self):
        return self.__values

    @staticmethod
    def decide_element():
        return "TestFeature"
