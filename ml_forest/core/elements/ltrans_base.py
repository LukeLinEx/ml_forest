from ml_forest.core.elements.identity import Base


class LTransform(Base):
    def __init__(self, **kwargs):
        """

        :param db: list of dictionaries
        :param filepaths: list of dictionaries
        """
        super(LTransform, self).__init__(**kwargs)
        self.__essentials = {}

    @staticmethod
    def decide_element():
        return "LTransform"