from ml_forest.core.elements.identity import Base


class LTransform(Base):
    def __init__(self):
        super(LTransform, self).__init__()
        self.__essentials = {}

    @staticmethod
    def decide_element():
        return "LTransform"

    def encode_whole(self, fed_y):
        raise NotImplementedError
